"""
run_all.py - Unified Time Series Classification Pipeline

This script consolidates experiments from:
1. Deep Learning Baselines
2. MultiRocket with GridSearch Optimization
3. TiRex Embedding + Linear Probing

It loads data once, applies unified preprocessing, registers all models, 
trains them, and outputs a standardized comparison table.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import time
import numpy as np
import pandas as pd
import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
# Try importing specialized libraries (with fallbacks for execution safety)
try:
    from aeon.transformations.collection.convolution_based import MultiRocket
except ImportError:
    MultiRocket = None
    print("Warning: 'aeon' not installed. MultiRocket will be skipped.")

try:
    from tirex.models.embedding import TiRexEmbedding
except ImportError:
    TiRexEmbedding = None
    print("Warning: 'tirex' not installed. TiRex embedding will be mocked/skipped.")

try:
    from mantis.architecture import MantisV1, MantisV2
    from mantis.trainer import MantisTrainer
except ImportError:
    MantisV1 = None
    MantisV2 = None
    MantisTrainer = None
    print("Warning: 'mantis-tsfm' not installed. Mantis models will be skipped.")


# =============================================================================
# 2. GLOBAL CONFIGURATION
# =============================================================================
CONFIG = {
    "seed": 42,
    "dataset_name": "LSST",     # Target dataset name
    "batch_size": 64,
    "epochs": 50,               # Default DL epochs
    "learning_rate": 1e-3,
    "val_size": 0.2,
    "norm_mode": "global",      # 'global' or 'channel-wise'
    "data_augmentation": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "metrics": ["accuracy", "macro_f1", "weighted_logloss"],
    "results_path": "all_experiments_results_globalnorm.csv",
    "early_stopping_patience": 8,
    "early_stopping_min_delta": 1e-4,
    "mantis_target_len": 512,
    
    # Model specific configs
    "multirocket_kernels": 6250, # Optimal value found in notebook
    "multirocket_c": 1.0,        # Tuned LogisticRegression C
}

# Set seeds for reproducibility
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])


# =============================================================================
# 3. METRICS & UTILS
# =============================================================================
def weighted_multi_logloss(
    true_classes,
    predictions,
    object_weights=None,
    class_weights=None,
    return_object_contributions=False,
):
    """
    Evaluate a weighted multi-class logloss function.

    Parameters
    ----------
    true_classes : `pandas.Series`
        A pandas series with the true class for each object
    predictions : `pandas.DataFrame`
        A pandas data frame with the predicted probabilities of each class for
        every object. There should be one column for each class.
    object_weights : dict (optional)
        The weights to use for each object. These are used to weight objects
        within a given class. The overall class weights will be normalized to
        the values set by class_weights. If not specified, flat weights are
        used.
    class_weights : dict (optional)
        The weights to use for each class. If not specified, flat weights are
        assumed for each class.
    return_object_contributions : bool (optional)
        If True, return a pandas Series with the individual contributions from
        each object. Otherwise, return the sum over all classes (default).
    """
    # ensure inputs are pandas objects
    if not isinstance(true_classes, pd.Series):
        true_classes = pd.Series(true_classes)

    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    object_loglosses = pd.Series(1e10 * np.ones(len(true_classes)), index=true_classes.index)

    sum_class_weights = 0.0

    for class_name in np.unique(true_classes):
        class_mask = true_classes == class_name
        class_count = np.sum(class_mask)

        if object_weights is not None:
            class_object_weights = object_weights[class_mask]
        else:
            class_object_weights = np.ones(class_count)

        if class_weights is not None:
            class_weight = class_weights.get(class_name, 1)
        else:
            class_weight = 1

        if class_weight == 0:
            # No weight for this class, ignore it.
            object_loglosses[class_mask] = 0
            continue

        if class_name not in predictions.columns:
            raise ValueError(
                f"No predictions available for class {class_name}! Either compute them or set the weight for that class to 0."
            )

        class_predictions = predictions[class_name][class_mask]

        # clip for numerical stability
        class_predictions = np.clip(class_predictions.astype(float), 1e-15, 1.0)

        class_loglosses = (
            -class_weight
            * class_object_weights
            * np.log(class_predictions)
            / np.sum(class_object_weights)
        )

        object_loglosses[class_mask] = class_loglosses

        sum_class_weights += class_weight

    # normalize by sum of class weights
    if sum_class_weights == 0:
        raise ValueError("Sum of class weights is zero; cannot normalize logloss")

    object_loglosses = object_loglosses / float(sum_class_weights)

    if return_object_contributions:
        return object_loglosses
    else:
        return float(np.sum(object_loglosses))


def to_mantis_input(X: np.ndarray, target_len: int) -> np.ndarray:
    """Prepare data for Mantis: (N, C, T) -> (N, C, target_len) float32."""
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_t = torch.tensor(X, dtype=torch.float32)
    X_resized = F.interpolate(X_t, size=target_len, mode="linear", align_corners=False)
    return X_resized.numpy()


class TimeSeriesScaler:
    """Standardizes 3D time series data (N, Channels, Length)"""
    def __init__(self, mode="instance-wise"):
        self.mode = mode
        # StandardScaler is only used for 'global' and 'channel-wise' modes
        self.scaler = StandardScaler()

    def _is_noop(self) -> bool:
        if self.mode is None:
            return True
        mode_value = str(self.mode).strip().lower()
        return mode_value in {"none", "noop", "no", "false", "0"}

    def fit_transform(self, X):
        if self._is_noop():
            return X
        N, C, L = X.shape
        if self.mode == "global":
            X_flat = X.reshape(-1, 1)
            X_scaled = self.scaler.fit_transform(X_flat)
            return X_scaled.reshape(N, C, L)
            
        elif self.mode == "channel-wise": 
            X_reshaped = X.transpose(0, 2, 1).reshape(-1, C)
            X_scaled = self.scaler.fit_transform(X_reshaped)
            return X_scaled.reshape(N, L, C).transpose(0, 2, 1)
            
        elif self.mode == "instance-wise":
            # Compute mean and std for each individual sequence (axis=2 is the Length dimension)
            means = np.mean(X, axis=2, keepdims=True)
            stds = np.std(X, axis=2, keepdims=True)
            # Add a tiny epsilon (1e-8) to prevent division by zero for completely flat signals
            return (X - means) / (stds + 1e-8)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


    def transform(self, X):
        if self._is_noop():
            return X
        N, C, L = X.shape
        if self.mode == "global":
            X_flat = X.reshape(-1, 1)
            X_scaled = self.scaler.transform(X_flat)
            return X_scaled.reshape(N, C, L)
            
        elif self.mode == "channel-wise":
            X_reshaped = X.transpose(0, 2, 1).reshape(-1, C)
            X_scaled = self.scaler.transform(X_reshaped)
            return X_scaled.reshape(N, L, C).transpose(0, 2, 1)
            
        elif self.mode == "instance-wise":
            # Transform is identical to fit_transform for instance-wise, 
            # as it relies only on the sequence itself, not on historical training data.
            means = np.mean(X, axis=2, keepdims=True)
            stds = np.std(X, axis=2, keepdims=True)
            return (X - means) / (stds + 1e-8)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# 4. DATA LOADING & PREPROCESSING
# =============================================================================
def load_and_preprocess_data():
    """
    Loads dataset, handles train/val/test splitting, encodes labels, 
    and applies standard scaling.
    """
    
    # --- REPLACE WITH ACTUAL tslearn / dataset loading ---
    from tslearn.datasets import UCR_UEA_datasets # type: ignore
    ds = UCR_UEA_datasets()
    X_temp, y_temp, X_test, y_test = ds.load_dataset("LSST")
    # Ensure channel-first layout (N, C, T) for scalers and DL models
    X_temp = np.transpose(X_temp, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    
    # Mock data generation (N_samples, Channels, Length)
    # -----------------------------------------------------

    # Encode labels
    label_encoder = LabelEncoder()
    y_temp = label_encoder.fit_transform(y_temp)
    y_test = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)

    val_ratio = CONFIG["val_size"]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=CONFIG["seed"], stratify=y_temp
    )

    # Scaling
    scaler = TimeSeriesScaler(mode=CONFIG["norm_mode"])
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"Dataset split: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes


# =============================================================================
# 5. MODEL DEFINITIONS & WRAPPERS
# =============================================================================

# --- A. Deep Learning Wrapper & Architecture ---
class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 64, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2  
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout)
        )
        self.fc = nn.Linear(base_channels * 2, num_classes)

    def forward(self, x):
        features = self.net(x).squeeze(-1)
        return self.fc(features)


class SimpleRNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2, bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.RNN(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, nonlinearity='relu', bidirectional=bidirectional, 
                          dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class SimpleLSTM(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_size: int = 128, num_layers: int = 2, bidirectional: bool = True, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=bidirectional, 
                            dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, (h_n, c_n) = self.lstm(x)

        if self.lstm.bidirectional:
            # Concat the last forward state and last backward state
            last_out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            last_out = h_n[-1, :, :]
            
        return self.fc(last_out)



class PyTorchModelWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a PyTorch model to behave like a scikit-learn classifier"""
    def __init__(self, model):
        self.model = model.to(CONFIG["device"])
        self.criterion = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["learning_rate"])
        self.best_state = None
        
    def _build_class_weighted_loss(self, y_train):
        classes, counts = np.unique(y_train, return_counts=True)
        total = counts.sum()
        weights = total / (len(classes) * counts)
        weight_tensor = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
        weight_tensor[classes] = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight_tensor.to(CONFIG["device"]))

    def _evaluate_val_loss(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X_val, dtype=torch.float32).to(CONFIG["device"])
            tensor_y = torch.tensor(y_val).to(CONFIG["device"])
            outputs = self.model(tensor_X)
            loss = self.criterion(outputs, tensor_y)
        self.model.train()
        return float(loss.item())

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.criterion = self._build_class_weighted_loss(y_train)
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

        best_val = None
        patience = CONFIG["early_stopping_patience"]
        min_delta = CONFIG["early_stopping_min_delta"]
        patience_left = patience
        
        self.model.train()
        for epoch in range(CONFIG["epochs"]):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(CONFIG["device"], dtype=torch.float32)
                batch_y = batch_y.to(CONFIG["device"])
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Simplified logging for verbosity control
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{CONFIG['epochs']}] Loss: {total_loss/len(loader):.4f}")

            if X_val is not None and y_val is not None:
                val_loss = self._evaluate_val_loss(X_val, y_val)
                if best_val is None or val_loss < (best_val - min_delta):
                    best_val = val_loss
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_left = patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32).to(CONFIG["device"])
            outputs = self.model(tensor_X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class MantisLinearProbeWrapper(BaseEstimator, ClassifierMixin):
    """Extracts Mantis embeddings (channel-wise mean pool) then fits a linear probe."""

    def __init__(
        self,
        classifier,
        version: str = "v1",
        target_len: int | None = None,
        device: str | None = None,
    ):
        if MantisTrainer is None:
            raise ValueError("Mantis is not available. Install 'mantis-tsfm'.")
        self.classifier = classifier
        self.version = version.lower()
        self.target_len = target_len or CONFIG["mantis_target_len"]
        self.device = device or CONFIG["device"]

        if self.version == "v1":
            self.network = MantisV1(device=self.device).from_pretrained("paris-noah/Mantis-8M")
        elif self.version == "v2":
            self.network = MantisV2(device=self.device).from_pretrained("paris-noah/MantisV2")
        else:
            raise ValueError(f"Unknown Mantis version: {version}")

        self.mantis = MantisTrainer(device=self.device, network=self.network)

    def _channelwise_embeddings(self, X: np.ndarray) -> np.ndarray:
        X_prep = to_mantis_input(X, self.target_len)  # (N, C, L)
        n_samples, n_channels, _ = X_prep.shape
        embeddings = []
        for channel_idx in range(n_channels):
            Xc = X_prep[:, channel_idx : channel_idx + 1, :]
            embeddings.append(self.mantis.transform(Xc))
        Zc = np.stack(embeddings, axis=1)
        return Zc.mean(axis=1)

    def fit(self, X, y, X_val=None, y_val=None, split_name="train"):
        X_emb = self._channelwise_embeddings(X)
        self.classifier.fit(X_emb, y)
        return self

    def predict_proba(self, X, split_name="test"):
        X_emb = self._channelwise_embeddings(X)
        return self.classifier.predict_proba(X_emb)

    def predict(self, X, split_name="test"):
        X_emb = self._channelwise_embeddings(X)
        return self.classifier.predict(X_emb)


class MultiRocketPipelineWrapper(BaseEstimator, ClassifierMixin):
    """Extracts MultiRocket features and uses a linear probe (no caching)."""
    
    def __init__(self, classifier, data_dir="data", n_kernels=6250, random_state=42):
        self.classifier = classifier
        self.data_dir = data_dir
        self.n_kernels = n_kernels
        self.random_state = random_state
        
        # Initialize MultiRocket only if aeon is installed
        if MultiRocket is not None:
            self.embedder = MultiRocket(n_kernels=self.n_kernels, random_state=self.random_state)
        else:
            self.embedder = None

    def _ensure_channel_first(self, X: np.ndarray) -> np.ndarray:
        """Ensure input is (N, C, T). If (N, T, C), swap axes."""
        if X.ndim == 3 and X.shape[1] > X.shape[2]:
            return np.transpose(X, (0, 2, 1))
        return X
            
    def fit(self, X, y, X_val=None, y_val=None, split_name="train"):
        print("Preparing MultiRocket features for training...")
        X = self._ensure_channel_first(X)
        if self.embedder is None:
            raise ValueError("Embedding cannot be None")
        X_emb = self.embedder.fit_transform(X)
        print("Fitting classifier...")
        self.classifier.fit(X_emb, y)
        return self

    def _get_features(self, X, split_name=None):
        X = self._ensure_channel_first(X)
        if self.embedder is None:
            raise ValueError("Embedding can not be none")
        return self.embedder.transform(X)

    def predict_proba(self, X, split_name="test"):
        X_emb = self._get_features(X, split_name=split_name)
        return self.classifier.predict_proba(X_emb)

    def predict(self, X, split_name="test"):
        X_emb = self._get_features(X, split_name=split_name)
        return self.classifier.predict(X_emb)


# =============================================================================
# 6. TRAINING PROCEDURE & EXPERIMENT RUNNER
# =============================================================================
def compute_metrics(y_true, y_pred, y_pred_proba):
    """Compute accuracy, macro F1, and weighted multi-class logloss."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    w_logloss = weighted_multi_logloss(y_true, y_pred_proba)
    return acc, f1, w_logloss


def fit_model(model, X_train, y_train, X_val=None, y_val=None, split_name="train"):
    """Fit a model, passing validation data if supported."""
    try:
        model.fit(X_train, y_train, X_val, y_val, split_name=split_name)
    except TypeError:
        # Fallback for models that don't accept split_name or val args
        try:
            model.fit(X_train, y_train, X_val, y_val)
        except TypeError:
            model.fit(X_train, y_train)


def evaluate_model(model, X, y, split_name="test"):
    """Predict and compute metrics on a given split."""
    try:
        y_pred = model.predict(X, split_name=split_name)
        y_pred_proba = model.predict_proba(X, split_name=split_name)
    except TypeError:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
    acc, f1, w_logloss = compute_metrics(y, y_pred, y_pred_proba)
    return acc, f1, w_logloss


def run_multirocket_experiments(train_data, val_data, test_data):
    """Run MultiRocket experiments mirroring the notebook (kernels, scaler, class_weight, C grid)."""
    results = []
    c_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    X_train_full = np.concatenate([train_data[0], val_data[0]], axis=0)
    y_train_full = np.concatenate([train_data[1], val_data[1]], axis=0)

    best_params = None
    best_val_score = None

    def make_classifier(c_value, use_scaler, use_balanced):
        class_weight = "balanced" if use_balanced else None
        clf = LogisticRegression(
            C=c_value,
            penalty="l2",
            max_iter=1000,
            class_weight=class_weight,
            random_state=CONFIG["seed"],
            solver="lbfgs",
        )
        if use_scaler:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("lr", clf),
            ])
        return clf

    def run_single(name, n_kernels, c_value, use_scaler=False, use_balanced=False):
        wrapper = MultiRocketPipelineWrapper(
            n_kernels=n_kernels,
            classifier=make_classifier(c_value, use_scaler, use_balanced),
        )
        fit_model(wrapper, train_data[0], train_data[1], val_data[0], val_data[1], split_name="train")
        val_acc, val_f1, val_wlogloss = evaluate_model(wrapper, val_data[0], val_data[1], split_name="val")
        test_acc, test_f1, test_wlogloss = evaluate_model(wrapper, test_data[0], test_data[1], split_name="test")
        nonlocal best_params, best_val_score
        if best_val_score is None or val_f1 > best_val_score:
            best_val_score = val_f1
            best_params = {
                "n_kernels": n_kernels,
                "c_value": c_value,
                "use_scaler": use_scaler,
                "use_balanced": use_balanced,
            }
        results.append({
            "Model": "MultiRocket",
            "Experiment": name,
            "Kernels": n_kernels,
            "C": c_value,
            "Scaler": use_scaler,
            "ClassWeight": "balanced" if use_balanced else "none",
            "Val_Accuracy": float(val_acc),
            "Val_Macro_F1": float(val_f1),
            "Val_WLogLoss": float(val_wlogloss),
            "Test_Accuracy": float(test_acc),
            "Test_Macro_F1": float(test_f1),
            "Test_WLogLoss": float(test_wlogloss),
        })

    # 1k kernels baseline + incremental improvements (notebook-style)
    run_single("[1] Baseline (L2, C=1, 1k kernels)", 1000, 1.0, use_scaler=False, use_balanced=False)
    run_single("[2] + class_weight='balanced'", 1000, 1.0, use_scaler=False, use_balanced=True)
    run_single("[3] + StandardScaler", 1000, 1.0, use_scaler=True, use_balanced=False)
    run_single("[4] + Scaler + class_weight='balanced'", 1000, 1.0, use_scaler=True, use_balanced=True)

    # 6.25k kernels baseline + full improvements
    run_single("[5] 6250 kernels (baseline C=1)", 6250, 1.0, use_scaler=False, use_balanced=False)
    run_single("[6] 6250 kernels + Scaler + balanced", 6250, 1.0, use_scaler=True, use_balanced=True)

    # GridSearch-style selection on C for 1k kernels (Scaler + balanced)
    best_c_1k = None
    best_val_1k = None
    for c_value in c_grid:
        wrapper = MultiRocketPipelineWrapper(
            n_kernels=1000,
            classifier=make_classifier(c_value, use_scaler=True, use_balanced=True),
        )
        fit_model(wrapper, train_data[0], train_data[1], val_data[0], val_data[1], split_name="train")
        _, val_f1, _ = evaluate_model(wrapper, val_data[0], val_data[1], split_name="val")
        if best_val_1k is None or val_f1 > best_val_1k:
            best_val_1k = val_f1
            best_c_1k = c_value
        if best_val_score is None or val_f1 > best_val_score:
            best_val_score = val_f1
            best_params = {
                "n_kernels": 1000,
                "c_value": c_value,
                "use_scaler": True,
                "use_balanced": True,
            }

    if best_c_1k is not None:
        # Retrain best config on train+val, then evaluate on test
        name = f"[7] GridSearch C={best_c_1k} (1k+Scaler+bal)"
        wrapper = MultiRocketPipelineWrapper(
            n_kernels=1000,
            classifier=make_classifier(best_c_1k, use_scaler=True, use_balanced=True),
        )
        fit_model(wrapper, X_train_full, y_train_full, split_name="trainval")
        test_acc, test_f1, test_wlogloss = evaluate_model(wrapper, test_data[0], test_data[1], split_name="test")
        results.append({
            "Model": "MultiRocket",
            "Experiment": name + " (retrain train+val)",
            "Kernels": 1000,
            "C": best_c_1k,
            "Scaler": True,
            "ClassWeight": "balanced",
            "Val_Accuracy": np.nan,
            "Val_Macro_F1": np.nan,
            "Val_WLogLoss": np.nan,
            "Test_Accuracy": float(test_acc),
            "Test_Macro_F1": float(test_f1),
            "Test_WLogLoss": float(test_wlogloss),
        })

    # GridSearch-style selection on C for 6.25k kernels (Scaler + balanced)
    best_c_6k = None
    best_val_6k = None
    for c_value in c_grid:
        wrapper = MultiRocketPipelineWrapper(
            n_kernels=6250,
            classifier=make_classifier(c_value, use_scaler=True, use_balanced=True),
        )
        fit_model(wrapper, train_data[0], train_data[1], val_data[0], val_data[1], split_name="train")
        _, val_f1, _ = evaluate_model(wrapper, val_data[0], val_data[1], split_name="val")
        if best_val_6k is None or val_f1 > best_val_6k:
            best_val_6k = val_f1
            best_c_6k = c_value
        if best_val_score is None or val_f1 > best_val_score:
            best_val_score = val_f1
            best_params = {
                "n_kernels": 6250,
                "c_value": c_value,
                "use_scaler": True,
                "use_balanced": True,
            }

    if best_c_6k is not None:
        # Retrain best config on train+val, then evaluate on test
        name = f"[8] GridSearch C={best_c_6k} (6k+Scaler+bal)"
        wrapper = MultiRocketPipelineWrapper(
            n_kernels=6250,
            classifier=make_classifier(best_c_6k, use_scaler=True, use_balanced=True),
        )
        fit_model(wrapper, X_train_full, y_train_full, split_name="trainval")
        test_acc, test_f1, test_wlogloss = evaluate_model(wrapper, test_data[0], test_data[1], split_name="test")
        results.append({
            "Model": "MultiRocket",
            "Experiment": name + " (retrain train+val)",
            "Kernels": 6250,
            "C": best_c_6k,
            "Scaler": True,
            "ClassWeight": "balanced",
            "Val_Accuracy": np.nan,
            "Val_Macro_F1": np.nan,
            "Val_WLogLoss": np.nan,
            "Test_Accuracy": float(test_acc),
            "Test_Macro_F1": float(test_f1),
            "Test_WLogLoss": float(test_wlogloss),
        })

    if best_params is not None:
        wrapper = MultiRocketPipelineWrapper(
            n_kernels=best_params["n_kernels"],
            classifier=make_classifier(
                best_params["c_value"],
                use_scaler=best_params["use_scaler"],
                use_balanced=best_params["use_balanced"],
            ),
        )
        fit_model(wrapper, X_train_full, y_train_full, split_name="trainval")
        test_acc, test_f1, test_wlogloss = evaluate_model(wrapper, test_data[0], test_data[1], split_name="test")
        results.append({
            "Model": "MultiRocket",
            "Experiment": "[FINAL] Best Val → retrain train+val",
            "Kernels": best_params["n_kernels"],
            "C": best_params["c_value"],
            "Scaler": best_params["use_scaler"],
            "ClassWeight": "balanced" if best_params["use_balanced"] else "none",
            "Val_Accuracy": np.nan,
            "Val_Macro_F1": float(best_val_score) if best_val_score is not None else np.nan,
            "Val_WLogLoss": np.nan,
            "Test_Accuracy": float(test_acc),
            "Test_Macro_F1": float(test_f1),
            "Test_WLogLoss": float(test_wlogloss),
        })

    return results


def main():
    print("=== STARTING UNIFIED EXPERIMENT PIPELINE ===")
    
    # 1. Load Data
    (train_data, val_data, test_data, num_classes) = load_and_preprocess_data()
    
    # 2. Register Models + simple hyperparameter grids
    in_channels = train_data[0].shape[1]

    model_candidates = {
        "Baseline_CNN1D": [
            ("cnn_baseline", PyTorchModelWrapper(SimpleCNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=64,
                kernel_size=3,
                dropout=0.3,
            ))),
            ("cnn_grid_1", PyTorchModelWrapper(SimpleCNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=128,
                kernel_size=5,
                dropout=0.3,
            ))),
            ("cnn_grid_2", PyTorchModelWrapper(SimpleCNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=64,
                kernel_size=5,
                dropout=0.5,
            ))),
        ],
        "Baseline_RNN1D": [
            ("rnn_baseline", PyTorchModelWrapper(SimpleRNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            ))),
            ("rnn_grid_1", PyTorchModelWrapper(SimpleRNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=64,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            ))),
            ("rnn_grid_2", PyTorchModelWrapper(SimpleRNN1D(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=1,
                bidirectional=False,
                dropout=0.2,
            ))),
        ],
        "Baseline_LSTM1D": [
            ("lstm_baseline", PyTorchModelWrapper(SimpleLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            ))),
            ("lstm_grid_1", PyTorchModelWrapper(SimpleLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=64,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            ))),
            ("lstm_grid_2", PyTorchModelWrapper(SimpleLSTM(
                in_channels=in_channels,
                num_classes=num_classes,
                hidden_size=128,
                num_layers=1,
                bidirectional=False,
                dropout=0.2,
            ))),
        ],
    }
    if MantisTrainer is not None:
        def make_mantis_probe(c_value, use_scaler, use_balanced):
            class_weight = "balanced" if use_balanced else None
            clf = LogisticRegression(
                C=c_value,
                max_iter=5000,
                solver="lbfgs",
                multi_class="multinomial",
                class_weight=class_weight,
                random_state=CONFIG["seed"],
            )
            if use_scaler:
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", clf),
                ])
            return clf

        mantis_grid = [
            ("c0.1_noscaler_unbalanced", 0.1, False, False),
            ("c1_noscaler_unbalanced", 1.0, False, False),
            ("c10_noscaler_unbalanced", 10.0, False, False),
            ("c1_scaler_unbalanced", 1.0, True, False),
            ("c1_scaler_balanced", 1.0, True, True),
            ("c10_scaler_balanced", 10.0, True, True),
        ]

        model_candidates["Mantis_V1_LinearProbe"] = [
            (
                f"mantis_v1_{tag}",
                MantisLinearProbeWrapper(
                    classifier=make_mantis_probe(c_value, use_scaler, use_balanced),
                    version="v1",
                ),
            )
            for tag, c_value, use_scaler, use_balanced in mantis_grid
        ]

        model_candidates["Mantis_V2_LinearProbe"] = [
            (
                f"mantis_v2_{tag}",
                MantisLinearProbeWrapper(
                    classifier=make_mantis_probe(c_value, use_scaler, use_balanced),
                    version="v2",
                ),
            )
            for tag, c_value, use_scaler, use_balanced in mantis_grid
        ]

    if MultiRocket is not None:
        model_candidates["MultiRocket_Optimized"] = []

    # 3. Run Experiments with validation-based selection
    results = []
    for group_name, candidates in model_candidates.items():
        if group_name == "MultiRocket_Optimized":
            multirocket_results = run_multirocket_experiments(train_data, val_data, test_data)
            results.extend(multirocket_results)
            pd.DataFrame(results).to_csv(CONFIG["results_path"], index=False)
            print(f"  Saved intermediate results to {CONFIG['results_path']}")
            continue
        print(f"\n=== TUNING {group_name} ===")
        best_val = None
        best_tag = None
        best_model = None

        for tag, model in candidates:
            print(f"\n[{group_name}] Training candidate: {tag}")
            start_time = time.time()
            fit_model(model, train_data[0], train_data[1], val_data[0], val_data[1], split_name="train")
            train_time = time.time() - start_time

            val_acc, val_f1, val_wlogloss = evaluate_model(model, val_data[0], val_data[1], split_name="val")
            print(f"  Val -> Acc: {val_acc:.4f} | F1: {val_f1:.4f} | W-LogLoss: {val_wlogloss:.4f}")

            test_acc, test_f1, test_wlogloss = evaluate_model(model, test_data[0], test_data[1], split_name="test")
            results.append({
                "Model": group_name,
                "Config": tag,
                "Val_Accuracy": float(val_acc),
                "Val_Macro_F1": float(val_f1),
                "Val_WLogLoss": float(val_wlogloss),
                "Test_Accuracy": float(test_acc),
                "Test_Macro_F1": float(test_f1),
                "Test_Weighted_LogLoss": float(test_wlogloss),
            })
            pd.DataFrame(results).to_csv(CONFIG["results_path"], index=False)
            print(f"  Saved intermediate results to {CONFIG['results_path']}")

            if best_val is None or best_val < val_f1:
                best_val = val_f1
                best_tag = tag
                best_model = model

        if best_model is None:
            print(f"No valid model for {group_name}")
            continue

        # Refit best model on train+val, then evaluate on test
        print(f"\n[{group_name}] Best config: {best_tag} (Val Macro F1: {best_val:.4f})")
        X_retrain = np.concatenate([train_data[0], val_data[0]], axis=0)
        y_retrain = np.concatenate([train_data[1], val_data[1]], axis=0)
        fit_model(best_model, X_retrain, y_retrain, split_name="trainval")

        test_acc, test_f1, test_wlogloss = evaluate_model(best_model, test_data[0], test_data[1], split_name="test")

        results.append({
            "Model": group_name,
            "Config": f"{best_tag} (retrain train+val)",
            "Val_Accuracy": np.nan,
            "Val_Macro_F1": float(best_val),
            "Val_WLogLoss": np.nan,
            "Test_Accuracy": float(test_acc),
            "Test_Macro_F1": float(test_f1),
            "Test_Weighted_LogLoss": float(test_wlogloss),
        })

        print(f"  Test -> Acc: {test_acc:.4f} | F1: {test_f1:.4f} | W-LogLoss: {test_wlogloss:.4f}")

        # Save results after each model group finishes to avoid data loss on crash
        pd.DataFrame(results).to_csv(CONFIG["results_path"], index=False)
        print(f"  Saved intermediate results to {CONFIG['results_path']}")

    # 4. Output Comparison
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save to CSV
    results_df.to_csv(CONFIG["results_path"], index=False)
    print(f"\nResults successfully saved to {CONFIG['results_path']}")


if __name__ == "__main__":
    main()