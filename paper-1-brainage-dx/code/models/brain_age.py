"""
BrainAge-Dx: Brain Age Prediction Models
Global + 7 regional Ridge Regression models.
"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    N_NETWORKS,
    N_FOLDS,
    RANDOM_STATE,
    NETWORK_NAMES,
    NETWORK_ABBREVS,
    RESULTS_DIR,
)


class BrainAgeModel:
    """
    Brain age prediction using Ridge Regression.
    Trains one global model (all 28 FC features) and
    7 regional models (per-network features).
    """

    def __init__(self, alphas=None):
        if alphas is None:
            alphas = np.logspace(-2, 4, 50)

        self.alphas = alphas
        self.global_model = None
        self.regional_models = {}
        self.global_scaler = StandardScaler()
        self.regional_scalers = {}
        self.bias_correction = {}  # For age-dependent bias correction

    def _get_global_features(self, df):
        """Extract all 28 FC features + sex."""
        feat_cols = [c for c in df.columns if c.startswith(("within_", "between_"))]
        X = df[feat_cols].values
        if "SEX" in df.columns:
            sex = df["SEX"].values.reshape(-1, 1)
            X = np.hstack([X, sex])
        return X

    def _get_regional_features(self, df, network_idx):
        """
        Extract features relevant to a specific network.
        Includes: within-network FC + all between-network FC involving this network.
        """
        abbrev = NETWORK_ABBREVS[network_idx]
        cols = []

        # Within-network
        within_col = f"within_{abbrev}"
        if within_col in df.columns:
            cols.append(within_col)

        # Between-network (this network paired with any other)
        for col in df.columns:
            if col.startswith("between_") and abbrev in col:
                cols.append(col)

        X = df[cols].values

        if "SEX" in df.columns:
            sex = df["SEX"].values.reshape(-1, 1)
            X = np.hstack([X, sex])

        return X

    def train(self, train_df):
        """
        Train global + 7 regional brain age models on healthy subjects.

        Args:
            train_df: DataFrame with FC features + 'AGE_AT_SCAN' + 'SEX'
        """
        y = train_df["AGE_AT_SCAN"].values.astype(float)

        print("Training Brain Age Models")
        print("=" * 50)
        print(f"Training subjects: {len(y)}")
        print(f"Age range: {y.min():.1f} - {y.max():.1f}")
        print()

        # ── Global Model ──
        X_global = self._get_global_features(train_df)
        X_global_scaled = self.global_scaler.fit_transform(X_global)

        self.global_model = RidgeCV(alphas=self.alphas, cv=N_FOLDS)
        self.global_model.fit(X_global_scaled, y)

        # CV predictions for bias correction
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        y_pred_cv = cross_val_predict(
            RidgeCV(alphas=self.alphas, cv=5),
            X_global_scaled,
            y,
            cv=cv,
        )

        mae_global = mean_absolute_error(y, y_pred_cv)
        r2_global = r2_score(y, y_pred_cv)

        # Bias correction: fit linear model of gap vs age
        gap_cv = y_pred_cv - y
        from sklearn.linear_model import LinearRegression

        bias_model = LinearRegression().fit(y.reshape(-1, 1), gap_cv)
        self.bias_correction["global"] = bias_model

        print(f"Global Model:")
        print(f"  MAE: {mae_global:.2f} years")
        print(f"  R²:  {r2_global:.3f}")
        print(f"  Best alpha: {self.global_model.alpha_:.2f}")
        print()

        # ── Regional Models ──
        results = {"model": [], "mae": [], "r2": [], "n_features": []}

        for net_idx in range(N_NETWORKS):
            X_reg = self._get_regional_features(train_df, net_idx)
            scaler = StandardScaler()
            X_reg_scaled = scaler.fit_transform(X_reg)
            self.regional_scalers[net_idx] = scaler

            model = RidgeCV(alphas=self.alphas, cv=N_FOLDS)
            model.fit(X_reg_scaled, y)
            self.regional_models[net_idx] = model

            # CV predictions
            y_pred_cv = cross_val_predict(
                RidgeCV(alphas=self.alphas, cv=5),
                X_reg_scaled,
                y,
                cv=cv,
            )

            mae = mean_absolute_error(y, y_pred_cv)
            r2 = r2_score(y, y_pred_cv)

            # Bias correction
            gap_cv = y_pred_cv - y
            bias_model = LinearRegression().fit(y.reshape(-1, 1), gap_cv)
            self.bias_correction[net_idx] = bias_model

            results["model"].append(NETWORK_NAMES[net_idx])
            results["mae"].append(mae)
            results["r2"].append(r2)
            results["n_features"].append(X_reg.shape[1])

            print(f"  {NETWORK_NAMES[net_idx]:20s}: MAE={mae:.2f}y, R²={r2:.3f} ({X_reg.shape[1]} features)")

        # Save training results
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / "training_results.csv", index=False)
        print(f"\nTraining results saved to {RESULTS_DIR / 'training_results.csv'}")

        return results_df

    def predict(self, test_df, apply_bias_correction=True):
        """
        Predict brain age and compute gaps for test subjects.

        Args:
            test_df: DataFrame with FC features + 'AGE_AT_SCAN' + 'SEX'
            apply_bias_correction: bool

        Returns:
            DataFrame with columns:
            - predicted_age_global, gap_global
            - predicted_age_{network}, gap_{network} for each of 7 networks
        """
        y_true = test_df["AGE_AT_SCAN"].values.astype(float)
        result = test_df.copy()

        # ── Global predictions ──
        X_global = self._get_global_features(test_df)
        X_global_scaled = self.global_scaler.transform(X_global)
        y_pred_global = self.global_model.predict(X_global_scaled)

        gap_global = y_pred_global - y_true
        if apply_bias_correction and "global" in self.bias_correction:
            correction = self.bias_correction["global"].predict(y_true.reshape(-1, 1))
            gap_global = gap_global - correction

        result["predicted_age_global"] = y_pred_global
        result["gap_global"] = gap_global

        # ── Regional predictions ──
        for net_idx in range(N_NETWORKS):
            abbrev = NETWORK_ABBREVS[net_idx]

            X_reg = self._get_regional_features(test_df, net_idx)
            X_reg_scaled = self.regional_scalers[net_idx].transform(X_reg)
            y_pred_reg = self.regional_models[net_idx].predict(X_reg_scaled)

            gap_reg = y_pred_reg - y_true
            if apply_bias_correction and net_idx in self.bias_correction:
                correction = self.bias_correction[net_idx].predict(y_true.reshape(-1, 1))
                gap_reg = gap_reg - correction

            result[f"predicted_age_{abbrev}"] = y_pred_reg
            result[f"gap_{abbrev}"] = gap_reg

        return result

    def save(self, path=None):
        """Save all models to disk."""
        if path is None:
            path = RESULTS_DIR / "brain_age_models.joblib"
        joblib.dump(
            {
                "global_model": self.global_model,
                "regional_models": self.regional_models,
                "global_scaler": self.global_scaler,
                "regional_scalers": self.regional_scalers,
                "bias_correction": self.bias_correction,
                "alphas": self.alphas,
            },
            path,
        )
        print(f"Models saved to {path}")

    def load(self, path=None):
        """Load models from disk."""
        if path is None:
            path = RESULTS_DIR / "brain_age_models.joblib"
        data = joblib.load(path)
        self.global_model = data["global_model"]
        self.regional_models = data["regional_models"]
        self.global_scaler = data["global_scaler"]
        self.regional_scalers = data["regional_scalers"]
        self.bias_correction = data["bias_correction"]
        self.alphas = data["alphas"]
        print(f"Models loaded from {path}")


if __name__ == "__main__":
    # Quick test with synthetic data
    print("BrainAge-Dx: Brain Age Model — Test Run")
    print("=" * 50)

    np.random.seed(RANDOM_STATE)

    # Generate synthetic training data (simulating HCP)
    n_train = 500
    feat_names = []
    for abbrev in NETWORK_ABBREVS:
        feat_names.append(f"within_{abbrev}")
    for i, j in combinations(range(N_NETWORKS), 2):
        feat_names.append(f"between_{NETWORK_ABBREVS[i]}_{NETWORK_ABBREVS[j]}")

    train_data = {name: np.random.randn(n_train) * 0.1 for name in feat_names}
    train_data["AGE_AT_SCAN"] = np.random.uniform(22, 35, n_train)
    train_data["SEX"] = np.random.choice([0, 1], n_train)

    # Make features weakly correlated with age
    for name in feat_names:
        train_data[name] += train_data["AGE_AT_SCAN"] * np.random.uniform(-0.01, 0.01)

    train_df = pd.DataFrame(train_data)

    # Train
    model = BrainAgeModel()
    model.train(train_df)

    # Generate synthetic test data
    n_test = 100
    test_data = {name: np.random.randn(n_test) * 0.1 for name in feat_names}
    test_data["AGE_AT_SCAN"] = np.random.uniform(10, 50, n_test)
    test_data["SEX"] = np.random.choice([0, 1], n_test)
    test_data["diagnosis"] = np.random.choice(["ASD", "HC", "SZ", "BD", "ADHD"], n_test)
    test_df = pd.DataFrame(test_data)

    # Predict
    results = model.predict(test_df)
    print(f"\nPrediction results shape: {results.shape}")
    print(f"Gap columns: {[c for c in results.columns if c.startswith('gap_')]}")

    # Save
    model.save()
    print("\nTest run complete!")
