import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import shap

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

plt.rcParams["font.family"] = "Arial"
# =============================================================================
# Global configuration
# =============================================================================
BASE_OUTPUT = "output"
os.makedirs(BASE_OUTPUT, exist_ok=True)

# =============================================================================
# Publication figure saving utility
# Save figures as PDF and TIFF formats only
# =============================================================================
def save_figure_pdf_tiff(fig, filepath):
    fig.savefig(
        filepath + ".pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.savefig(
        filepath + ".tiff",
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.1,
    )



DATA_PATH = r"D:\Projects\PythonProjects\PythonProject1\data\alzheimers_disease_data.csv"
TARGET_COL = "Diagnosis"

RANDOM_STATE = 42
CV_FOLDS = 10
N_JOBS = max(1, min(8, os.cpu_count() or 1))
PSO_PARTICLES = 10
PSO_ITERATIONS = 15


LASSO_FIXED_ALPHA = None
LASSO_INNER_CV_FOLDS = 10


# =============================================================================
# Utility: leakage-safe LASSO feature selector
# =============================================================================
class LassoFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        alpha=None,
        inner_cv_folds=10,
        random_state=42,
        max_iter=10000,
        coef_tol=1e-12,
    ):
        self.alpha = alpha
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state
        self.max_iter = max_iter
        self.coef_tol = coef_tol

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.alpha is None:
            inner_cv = StratifiedKFold(
                n_splits=self.inner_cv_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            self.selector_model_ = LassoCV(
                cv=inner_cv,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_jobs=1,  # avoid nested parallelism
            )
        else:
            self.selector_model_ = Lasso(
                alpha=float(self.alpha),
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        self.selector_model_.fit(X_arr, y_arr)
        coef = np.asarray(self.selector_model_.coef_)
        self.support_ = np.abs(coef) > self.coef_tol

        # Safety fallback: never return zero features.
        if not np.any(self.support_):
            self.support_[int(np.argmax(np.abs(coef)))] = True

        self.n_features_in_ = X_arr.shape[1]
        self.alpha_ = (
            float(self.selector_model_.alpha_)
            if hasattr(self.selector_model_, "alpha_")
            else float(self.alpha)
        )
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        return X_arr[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()


# =============================================================================
# Utility: CV splitter
# =============================================================================
def make_outer_cv():
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# =============================================================================
# Utility: construct leakage-safe pipeline
# =============================================================================
def build_pipeline(classifier):
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoFeatureSelector(
                    alpha=LASSO_FIXED_ALPHA,
                    inner_cv_folds=LASSO_INNER_CV_FOLDS,
                    random_state=RANDOM_STATE,
                    max_iter=10000,
                ),
            ),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


# =============================================================================
# Simplified CPSO with particle-level parallelism
# =============================================================================
class SimplePSO:
    def __init__(
        self,
        objective_func,
        bounds,
        n_particles=10,
        max_iter=20,
        w=0.7,
        c1=2.0,
        c2=2.0,
        n_jobs=1,
        random_state=42,
    ):
        self.objective = objective_func
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.dim = len(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _logistic_sequence(x0, length, mu=4.0):
        seq = []
        x = float(x0)
        for _ in range(length):
            x = mu * x * (1.0 - x)
            seq.append(x)
        return np.asarray(seq)

    def _evaluate_positions(self, positions):
        return np.asarray(
            Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self.objective)(p.copy()) for p in positions
            )
        )

    def optimize(self):
        # ---------------- Chaotic initialization ----------------
        positions = np.zeros((self.n_particles, self.dim), dtype=float)
        base_x0 = 0.1
        for i in range(self.n_particles):
            x0 = (base_x0 + i * 0.01) % 1.0
            if np.isclose(x0, 0.0):
                x0 = 0.001
            chaos_seq = self._logistic_sequence(x0, self.dim)
            positions[i] = self.lb + chaos_seq * (self.ub - self.lb)

        velocities = self.rng.uniform(
            low=-(self.ub - self.lb) / 2.0,
            high=(self.ub - self.lb) / 2.0,
            size=(self.n_particles, self.dim),
        )

        # ---------------- Chaotic states for r1 and r2 ----------------
        mu = 4.0
        chaos_states_r1 = np.zeros((self.n_particles, self.dim), dtype=float)
        chaos_states_r2 = np.zeros((self.n_particles, self.dim), dtype=float)

        for i in range(self.n_particles):
            for j in range(self.dim):
                init1 = (0.1 + i * 0.01 + j * 0.001) % 1.0
                init2 = (0.2 + i * 0.01 + j * 0.001) % 1.0
                if np.isclose(init1, 0.0):
                    init1 = 0.001
                if np.isclose(init2, 0.0):
                    init2 = 0.001
                chaos_states_r1[i, j] = init1
                chaos_states_r2[i, j] = init2

        personal_best = positions.copy()
        personal_best_score = self._evaluate_positions(positions)

        best_idx = int(np.argmax(personal_best_score))
        global_best = personal_best[best_idx].copy()
        global_best_score = float(personal_best_score[best_idx])

        # ---------------- Main CPSO loop ----------------
        for iteration in range(self.max_iter):
            r1 = chaos_states_r1.copy()
            r2 = chaos_states_r2.copy()

            chaos_states_r1 = mu * chaos_states_r1 * (1.0 - chaos_states_r1)
            chaos_states_r2 = mu * chaos_states_r2 * (1.0 - chaos_states_r2)

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best - positions)
                + self.c2 * r2 * (global_best - positions)
            )

            positions = np.clip(positions + velocities, self.lb, self.ub)
            scores = self._evaluate_positions(positions)

            improve = scores > personal_best_score
            personal_best[improve] = positions[improve]
            personal_best_score[improve] = scores[improve]

            iter_best_idx = int(np.argmax(scores))
            iter_best_score = float(scores[iter_best_idx])
            if iter_best_score > global_best_score:
                global_best = positions[iter_best_idx].copy()
                global_best_score = iter_best_score

            print(
                f"    CPSO iteration {iteration + 1:02d}/{self.max_iter}: "
                f"best CV AUC = {global_best_score:.4f}"
            )

        return global_best, global_best_score


# =============================================================================
# Grid + CPSO optimization
# =============================================================================
def _is_numeric_grid(values):
    if len(values) == 0:
        return False
    for value in values:
        if isinstance(value, (bool, np.bool_)) or value is None:
            return False
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        if not np.isfinite(float(value)):
            return False
    return True


def _grid_is_integer(values):
    return all(
        isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
        for v in values
    )


def grid_cpso_optimize(
    model_name,
    base_pipeline,
    param_grid,
    X,
    y,
    n_iter_cpso=15,
):
    print(f"\n>>> {model_name}: GridSearchCV")
    grid_start = time.perf_counter()

    # One parallel layer here: GridSearchCV parallelizes candidate/fold fits.
    grid = GridSearchCV(
        estimator=clone(base_pipeline),
        param_grid=param_grid,
        cv=make_outer_cv(),
        scoring="roc_auc",
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    grid.fit(X, y)

    grid_seconds = time.perf_counter() - grid_start
    best_params_grid = grid.best_params_.copy()
    print(f"{model_name} grid-search best CV AUC: {grid.best_score_:.4f}")
    print(f"{model_name} grid-search time: {grid_seconds:.2f} s")

    numeric_params = [
        key for key, values in param_grid.items() if _is_numeric_grid(values)
    ]

    # If there is no purely numeric parameter to refine, return grid result.
    if not numeric_params:
        return {
            "best_params": best_params_grid,
            "grid_best_auc": float(grid.best_score_),
            "cpso_best_auc": float(grid.best_score_),
            "grid_seconds": grid_seconds,
            "cpso_seconds": 0.0,
        }

    # ---------------- Build local CPSO search bounds ----------------
    bounds = []
    param_keys = []

    for key in numeric_params:
        values = param_grid[key]
        low = float(min(values))
        high = float(max(values))
        best_val = float(best_params_grid[key])

        span = high - low
        if np.isclose(span, 0.0):
            continue

        new_low = max(low, best_val - 0.2 * span)
        new_high = min(high, best_val + 0.2 * span)

        if np.isclose(new_low, new_high):
            continue

        bounds.append((new_low, new_high))
        param_keys.append(key)

    if not param_keys:
        return {
            "best_params": best_params_grid,
            "grid_best_auc": float(grid.best_score_),
            "cpso_best_auc": float(grid.best_score_),
            "grid_seconds": grid_seconds,
            "cpso_seconds": 0.0,
        }

    # ---------------- CPSO objective ----------------
    def objective(vec):
        params = best_params_grid.copy()

        for i, key in enumerate(param_keys):
            val = float(vec[i])
            if _grid_is_integer(param_grid[key]):
                val = int(round(val))
            params[key] = val

        # Critical fix: clone a fresh pipeline for every particle evaluation.
        candidate_pipeline = clone(base_pipeline)
        candidate_pipeline.set_params(**params)

        # Inner to particle evaluation is deliberately single-process.
        # This avoids nested joblib parallelism.
        scores = cross_val_score(
            candidate_pipeline,
            X,
            y,
            cv=make_outer_cv(),
            scoring="roc_auc",
            n_jobs=1,
            error_score="raise",
        )
        return float(np.mean(scores))

    # ---------------- Run CPSO ----------------
    print(f">>> {model_name}: CPSO local refinement")
    cpso_start = time.perf_counter()

    pso = SimplePSO(
        objective_func=objective,
        bounds=bounds,
        n_particles=PSO_PARTICLES,
        max_iter=n_iter_cpso,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    best_vec, best_score = pso.optimize()

    cpso_seconds = time.perf_counter() - cpso_start
    print(f"{model_name} CPSO best CV AUC: {best_score:.4f}")
    print(f"{model_name} CPSO time: {cpso_seconds:.2f} s")

    final_params = best_params_grid.copy()
    for i, key in enumerate(param_keys):
        val = float(best_vec[i])
        if _grid_is_integer(param_grid[key]):
            val = int(round(val))
        final_params[key] = val

    return {
        "best_params": final_params,
        "grid_best_auc": float(grid.best_score_),
        "cpso_best_auc": float(best_score),
        "grid_seconds": grid_seconds,
        "cpso_seconds": cpso_seconds,
    }



# =============================================================================
# Model-performance visualization from saved result table
# =============================================================================
def plot_model_performance_from_csv(
    csv_path,
    output_pdf,
):
    performance_df = pd.read_csv(csv_path)

    required_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "G-mean",
        "AUC",
    ]
    missing_columns = [
        c for c in required_columns
        if c not in performance_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in model-performance table: "
            f"{missing_columns}"
        )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "G-mean",
        "AUC",
    ]
    models = performance_df["Model"].tolist()

    # Same layout and overall figure size as the previous visualization.
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 10),
    )
    plt.subplots_adjust(top=0.92)
    axes = axes.flatten()

    fill_base = 0.2

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = performance_df[metric].to_numpy(dtype=float)
        x = np.arange(len(models))

        # Same line, marker, and color.
        ax.plot(
            x,
            values,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=7,
            color="#f4cae4",
            label=metric,
        )

        # Same filled area.
        ax.fill_between(
            x,
            values,
            fill_base,
            color="#f4cae4",
            alpha=0.25,
        )

        # Same four-decimal value labels.
        for i, value in enumerate(values):
            ax.text(
                i,
                value + 0.008,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="medium",
            )

        ax.set_title(
            metric,
            fontsize=15,
            fontweight="bold",
        )
        ax.set_ylim(fill_base, 1.1)
        ax.set_ylabel(
            "Score",
            fontsize=15,
        )
        ax.tick_params(
            axis="y",
            labelsize=12
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            models,
            rotation=45,
            ha="right",
            fontsize=15,
        )

        # Same y-grid style.
        ax.grid(
            True,
            linestyle="--",
            alpha=0.5,
            axis="y",
        )

        # Same reference lines.
        for yref in [0.7, 0.8, 0.9]:
            ax.axhline(
                y=yref,
                color="gray",
                linestyle=":",
                alpha=0.3,
                linewidth=0.8,
            )


    plt.tight_layout()

    fig.savefig(
        output_pdf,
        format="pdf",
        bbox_inches="tight",
    )

    output_tiff = output_pdf.replace(
        ".pdf",
        ".tiff"
    )

    fig.savefig(
        output_tiff,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Model-performance visualization saved to: "
        f"{output_pdf}"
    )

# =============================================================================
# SHAP helpers
# =============================================================================
def transform_without_sampler(fitted_pipeline, X):
    """Apply fitted scaler and LASSO selector only; never apply SMOTE at inference."""
    Xt = fitted_pipeline.named_steps["scaler"].transform(X)
    Xt = fitted_pipeline.named_steps["lasso"].transform(Xt)
    return Xt


def perform_shap_analysis(
    fitted_pipeline,
    X_train_raw,
    X_test_raw,
    original_feature_names,
    model_name,
    background_size=100,
    explain_size=100,
    output_dir="shap_plots",
):
    full_output_dir = os.path.join(BASE_OUTPUT, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    classifier = fitted_pipeline.named_steps["classifier"]
    selector = fitted_pipeline.named_steps["lasso"]
    support = selector.get_support()
    selected_feature_names = np.asarray(original_feature_names)[support]

    X_train_model = transform_without_sampler(fitted_pipeline, X_train_raw)
    X_test_model = transform_without_sampler(fitted_pipeline, X_test_raw)

    X_train_df = pd.DataFrame(X_train_model, columns=selected_feature_names)
    X_test_df = pd.DataFrame(X_test_model, columns=selected_feature_names)

    if len(X_train_df) > background_size:
        X_train_sample = X_train_df.sample(n=background_size, random_state=RANDOM_STATE)
    else:
        X_train_sample = X_train_df

    if len(X_test_df) > explain_size:
        X_explain_sample = X_test_df.sample(n=explain_size, random_state=RANDOM_STATE)
    else:
        X_explain_sample = X_test_df

    model_class = classifier.__class__.__name__
    tree_models = {
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "ExtraTreesClassifier",
        "DecisionTreeClassifier",
    }

    print(f"SHAP model type: {model_class}")

    if model_class in tree_models:
        print("Using TreeExplainer...")
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_explain_sample)
    else:
        print("Using KernelExplainer (may be slow)...")
        explainer = shap.KernelExplainer(classifier.predict_proba, X_train_sample)
        shap_values = explainer.shap_values(X_explain_sample)

    # Binary-class SHAP compatibility across SHAP versions.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_explain_sample,
        feature_names=selected_feature_names,
        show=False,
    )

    fig = plt.gcf()
    ax = plt.gca()

    fig.set_size_inches(12, 8)

    ax.set_title(f"SHAP Summary Plot - {model_name}", fontsize=15)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=15)

    fig.subplots_adjust(
        left=0.30,
        right=0.95,
        bottom=0.14,
        top=0.92,
    )

    plt.savefig(
        os.path.join(
            full_output_dir,
            f"shap_summary_{model_name}.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.savefig(
        os.path.join(
            full_output_dir,
            f"shap_summary_{model_name}.tiff"
        ),
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_explain_sample,
        feature_names=selected_feature_names,
        plot_type="bar",
        show=False,
    )

    fig = plt.gcf()
    ax = plt.gca()

    fig.set_size_inches(12, 8)

    ax.set_title(f"SHAP Feature Importance - {model_name}", fontsize=15)
    ax.set_xlabel(
        "mean(|SHAP value|)\n(average impact on model output magnitude)",
        fontsize=15,
        labelpad=8,
    )

    fig.subplots_adjust(
        left=0.30,
        right=0.95,
        bottom=0.16,
        top=0.92,
    )

    save_figure_pdf_tiff(
        fig,
        os.path.join(full_output_dir, f"shap_bar_{model_name}")
    )

    plt.close(fig)

    shap_df = pd.DataFrame(shap_values, columns=selected_feature_names)
    shap_df.to_csv(
        os.path.join(full_output_dir, f"shap_values_{model_name}.csv"),
        index=False,
    )

    print(f"SHAP analysis saved to: {full_output_dir}")



# =============================================================================
# Descriptive visualization of final selected features
# =============================================================================
def plot_selected_feature_descriptions(
    df,
    selected_features,
    target_col="Diagnosis",
    output_dir="descriptive_analysis",
):
    full_output_dir = os.path.join(BASE_OUTPUT, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    selected_features = list(selected_features)
    if not selected_features:
        print("No selected features available; descriptive plot skipped.")
        return

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing from dataset: {missing}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Expected 12-feature grouping from the current final model.
    expected_continuous = [
        "Age",
        "DiastolicBP",
        "CholesterolLDL",
        "CholesterolHDL",
        "CholesterolTriglycerides",
        "MMSE",
        "FunctionalAssessment",
        "ADL",
    ]
    expected_discrete = [
        "Ethnicity",
        "FamilyHistoryAlzheimers",
        "MemoryComplaints",
        "BehavioralProblems",
    ]

    continuous_features = [f for f in expected_continuous if f in selected_features]
    discrete_features = [f for f in expected_discrete if f in selected_features]

    known = set(continuous_features + discrete_features)
    unexpected = [f for f in selected_features if f not in known]
    if unexpected:
        print(
            "Note: unexpected selected features will be plotted as continuous:",
            unexpected,
        )
        continuous_features.extend(unexpected)

    all_features = continuous_features + discrete_features

    if len(all_features) != 12:
        print(
            f"Note: {len(all_features)} selected features will be plotted     "
            "instead of 12."
        )

    # Continuous variables: standardize over the training set for descriptive display only.
    scaled_cont_df = None
    if continuous_features:
        descriptive_scaler = StandardScaler()
        scaled_continuous = descriptive_scaler.fit_transform(
            df[continuous_features]
        )
        scaled_cont_df = pd.DataFrame(
            scaled_continuous,
            columns=continuous_features,
            index=df.index,
        )
        scaled_cont_df[target_col] = df[target_col].to_numpy()

        scaled_cont_df.to_csv(
            os.path.join(
                full_output_dir,
                "selected_continuous_features_standardized_training_set.csv",
            ),
            index=False,
        )

    # Save the raw selected-feature table used by the descriptive figure.
    df[selected_features + [target_col]].to_csv(
        os.path.join(
            full_output_dir,
            "selected_features_raw_training_set.csv",
        ),
        index=False,
    )

    # 12 features -> 4 rows x 3 columns.
    n_features = len(all_features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig = plt.figure(figsize=(15, 4.0 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        hspace=0.25,
        wspace=0.25,
    )

    palette = {
        0: "#72ccff",
        1: "#f7f494",
    }

    ethnicity_color_map = {
        0: "#72ccff",
        1: "#f7f494",
        2: "#d2f5a6",
        3: "#76f2f2",
    }

    for idx, feature in enumerate(all_features):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        if feature in continuous_features:
            data_subset = scaled_cont_df[[feature, target_col]].rename(
                columns={feature: "Standardized Value"}
            )

            sns.boxplot(
                data=data_subset,
                x=target_col,
                y="Standardized Value",
                hue=target_col,
                palette=palette,
                dodge=False,
                width=0.6,
                linewidth=1,
                fliersize=0,
                ax=ax,
                legend=False,
            )

            sns.swarmplot(
                data=data_subset,
                x=target_col,
                y="Standardized Value",
                hue=target_col,
                palette=palette,
                dodge=False,
                size=2,
                alpha=0.7,
                ax=ax,
                legend=False,
            )

            ax.set_ylabel("Standardized Value", fontsize=15,fontweight="bold")
            ax.axhline(
                y=0,
                color="gray",
                linestyle=":",
                alpha=0.5,
                linewidth=0.8,
            )

        else:
            cross_tab = (
                pd.crosstab(
                    df[target_col],
                    df[feature],
                    normalize="index",
                )
                * 100.0
            )
            cross_tab = cross_tab.reindex(index=[0, 1], fill_value=0.0)
            categories = cross_tab.columns.tolist()

            if feature == "Ethnicity":
                bar_colors = [
                    ethnicity_color_map.get(category, "#cccccc")
                    for category in categories
                ]
            elif len(categories) == 2:
                bar_colors = ["#72ccff", "#f7f494"]
            else:
                bar_colors = list(
                    sns.color_palette("Set2", len(categories))
                )

            cross_tab.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=bar_colors,
                edgecolor="black",
                linewidth=0.5,
                legend=False,
            )

            ax.set_ylabel("Percentage (%)", fontsize=15,fontweight="bold")
            ax.set_ylim(0, 100)

            for container in ax.containers:
                labels = []
                for patch in container:
                    h = patch.get_height()
                    labels.append(f"{h:.1f}%" if h >= 4.0 else "")
                ax.bar_label(
                    container,
                    labels=labels,
                    fontsize=15,
                    label_type="center",
                )

        # Common style.
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Non-AD (0)", "AD (1)"],
            rotation=0,
            fontsize=15,
            fontweight="bold",
        )

        # Feature name is the subplot title; no (a), (b), ... labels.
        ax.set_title(
            feature,
            fontsize=15,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused cells if the selected-feature count changes.
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax_unused = fig.add_subplot(gs[row, col])
        ax_unused.axis("off")

    plt.subplots_adjust(
        left=0.06,
        right=0.94,
        top=0.96,
        bottom=0.06,
    )

    pdf_path = os.path.join(
        full_output_dir,
        "selected_feature_descriptive_distribution.pdf",
    )

    tiff_path = os.path.join(
        full_output_dir,
        "selected_feature_descriptive_distribution.tiff",
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.10,
    )

    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)

    print(
        "Selected-feature descriptive analysis saved to: "
        f"{full_output_dir}"
    )

# =============================================================================
# Conformal prediction helpers
# =============================================================================
def compute_entropy(proba, eps=1e-12):
    """Shannon entropy for any number of classes."""
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    return -np.sum(p * np.log(p), axis=1)


def build_normalized_mondrian_calibration(
    fitted_pipeline,
    X_cal,
    y_cal,
):
    cal_proba = fitted_pipeline.predict_proba(X_cal)
    classes = np.asarray(fitted_pipeline.named_steps["classifier"].classes_)
    class_to_col = {cls: idx for idx, cls in enumerate(classes)}

    cal_entropy = compute_entropy(cal_proba)
    cal_sigma = cal_entropy + 1e-6
    y_cal_arr = np.asarray(y_cal)

    cal_scores_norm = {}
    for cls in classes:
        col = class_to_col[cls]
        mask = y_cal_arr == cls
        raw_scores = 1.0 - cal_proba[mask, col]
        sigma_cls = cal_sigma[mask]
        cal_scores_norm[cls] = np.sort(raw_scores / sigma_cls)

    return classes, class_to_col, cal_scores_norm


def compute_mondrian_pvalues(
    test_proba,
    test_sigma,
    classes,
    class_to_col,
    cal_scores_norm,
):
    n = test_proba.shape[0]
    pvalues = np.zeros((n, len(classes)), dtype=float)

    for i in range(n):
        for j, cls in enumerate(classes):
            col = class_to_col[cls]
            raw_score = 1.0 - test_proba[i, col]
            norm_score = raw_score / test_sigma[i]
            cal_scores = cal_scores_norm[cls]
            pvalues[i, j] = (
                np.sum(cal_scores >= norm_score) + 1.0
            ) / (len(cal_scores) + 1.0)

    return pvalues


def prediction_sets_from_pvalues(pvalues, classes, alpha):
    result = []
    for row in pvalues:
        result.append([cls for cls, p in zip(classes, row) if p > alpha])
    return result


# =============================================================================
# Main workflow
# =============================================================================
def main():
    start_time = time.perf_counter()
    np.random.seed(RANDOM_STATE)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Optional guard against obvious non-numeric columns.
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "The current pipeline expects numeric predictors. "
            f"Non-numeric columns found: {non_numeric}"
        )

    # -------------------------------------------------------------------------
    # 2. Fixed 60/20/20 split
    # -------------------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Reset indices to simplify downstream indexing.
    X_train = X_train.reset_index(drop=True)
    X_cal = X_cal.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_cal = y_cal.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("=" * 70)
    print("Dataset split")
    print(f"Training set:    {len(X_train)} samples")
    print(f"Calibration set: {len(X_cal)} samples")
    print(f"Test set:        {len(X_test)} samples")
    print("Training distribution:\n", y_train.value_counts().sort_index())
    print("Calibration distribution:\n", y_cal.value_counts().sort_index())
    print("Test distribution:\n", y_test.value_counts().sort_index())

    split_data_dir = os.path.join(BASE_OUTPUT, "split_data")
    os.makedirs(split_data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(split_data_dir, "X_train.csv"), index=False)
    X_cal.to_csv(os.path.join(split_data_dir, "X_cal.csv"), index=False)
    X_test.to_csv(os.path.join(split_data_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(split_data_dir, "y_train.csv"), index=False)
    y_cal.to_csv(os.path.join(split_data_dir, "y_cal.csv"), index=False)
    y_test.to_csv(os.path.join(split_data_dir, "y_test.csv"), index=False)

    # -------------------------------------------------------------------------
    # 3. Define base classifiers
    #    Internal estimator-level parallelism is kept at 1 where applicable.
    # -------------------------------------------------------------------------
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "MLP": MLPClassifier(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "GBDT": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }

    # Pipeline parameter names use classifier__ prefix.
    param_grids = {
        "LogisticRegression": {
            "classifier__C": [0.01, 0.1, 1, 10],
        },
        "RandomForest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 10, None],
        },
        "MLP": {
            "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "classifier__alpha": [0.0001, 0.001, 0.01],
        },
        "DecisionTree": {
            "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "GBDT": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5],
        },
        "XGBoost": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__max_depth": [3, 5, 7],
        },
        "ExtraTrees": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, 10, None],
        },
        "SVM": {
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale", "auto", 0.1, 0.01],
            "classifier__kernel": ["rbf"],
        },
    }

    # -------------------------------------------------------------------------
    # 4. Leakage-safe Grid-CPSO optimization
    #    Models are processed sequentially; GridSearch/CPSO use parallelism inside.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Leakage-safe model optimization starts (N_JOBS={N_JOBS})")

    fitted_models = {}
    cv_scores = {}
    runtime_rows = []

    for name, classifier in models.items():
        model_start = time.perf_counter()
        base_pipeline = build_pipeline(classifier)

        opt_result = grid_cpso_optimize(
            model_name=name,
            base_pipeline=base_pipeline,
            param_grid=param_grids[name],
            X=X_train,
            y=y_train,
            n_iter_cpso=PSO_ITERATIONS,
        )

        final_pipeline = clone(base_pipeline)
        final_pipeline.set_params(**opt_result["best_params"])

        # Recompute final CV score with identical leakage-safe pipeline.
        cv_start = time.perf_counter()
        scores = cross_val_score(
            final_pipeline,
            X_train,
            y_train,
            cv=make_outer_cv(),
            scoring="roc_auc",
            n_jobs=N_JOBS,
            error_score="raise",
        )
        final_cv_auc = float(np.mean(scores))
        final_cv_sd = float(np.std(scores, ddof=1))
        cv_seconds = time.perf_counter() - cv_start

        # Final fit uses the full 60% training set only.
        fit_start = time.perf_counter()
        final_pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        fitted_models[name] = final_pipeline
        cv_scores[name] = final_cv_auc

        total_seconds = time.perf_counter() - model_start
        runtime_rows.append(
            {
                "Model": name,
                "Grid_best_CV_AUC": opt_result["grid_best_auc"],
                "CPSO_best_CV_AUC": opt_result["cpso_best_auc"],
                "Final_CV_AUC_mean": final_cv_auc,
                "Final_CV_AUC_sd": final_cv_sd,
                "Grid_seconds": opt_result["grid_seconds"],
                "CPSO_seconds": opt_result["cpso_seconds"],
                "Final_CV_seconds": cv_seconds,
                "Final_fit_seconds": fit_seconds,
                "Total_model_seconds": total_seconds,
                "N_JOBS": N_JOBS,
            }
        )

        print(f"{name} final leakage-safe CV AUC: {final_cv_auc:.4f} ± {final_cv_sd:.4f}")

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(
        os.path.join(BASE_OUTPUT, "optimization_runtime.csv"),
        index=False,
    )

    # Select the optimal model BEFORE looking at test-set metrics.
    best_name = max(cv_scores, key=cv_scores.get)
    best_pipeline = fitted_models[best_name]

    print("\n" + "=" * 70)
    print(f"Selected model based only on training-set CV AUC: {best_name}")
    print(f"Best CV AUC: {cv_scores[best_name]:.4f}")

    # -------------------------------------------------------------------------
    # 5. Save final selected features from the fitted best pipeline
    # -------------------------------------------------------------------------
    best_selector = best_pipeline.named_steps["lasso"]
    selected_mask = best_selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    print(f"Final LASSO alpha: {best_selector.alpha_:.10g}")
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:", selected_features)

    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        os.path.join(BASE_OUTPUT, "selected_features.csv"),
        index=False,
    )

    # -------------------------------------------------------------------------
    # 6. Final independent test evaluation of all preselected pipelines
    # -------------------------------------------------------------------------
    roc_curves_dir = os.path.join(BASE_OUTPUT, "roc_curves")
    os.makedirs(roc_curves_dir, exist_ok=True)

    metrics_list = []

    # 8 ROC subplots: 4 rows × 2 columns
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(11, 14),
    )
    axes = axes.flatten()

    colors = [
        "#fc97af",
        "#87f7cf",
        "#f7f494",
        "#72ccff",
        "#f7c5a0",
        "#d4a4eb",
        "#d2f5a6",
        "#76f2f2",
    ]

    # Short names used in the manuscript
    display_names = {
        "LogisticRegression": "LR",
        "RandomForest": "RF",
        "MLP": "MLP",
        "DecisionTree": "DT",
        "GBDT": "GBDT",
        "XGBoost": "XGBoost",
        "ExtraTrees": "ERT",
        "SVM": "SVM",
    }

    for idx, (name, pipeline) in enumerate(fitted_models.items()):
        # -------------------------------------------------------------
        # Independent test-set evaluation
        # -------------------------------------------------------------
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        gmean = (
            np.sqrt(rec * specificity)
            if np.isfinite(specificity)
            else np.nan
        )

        auc = roc_auc_score(y_test, y_proba)

        metrics_list.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "G-mean": gmean,
                "AUC": auc,
            }
        )

        # -------------------------------------------------------------
        # ROC curve for each model
        # -------------------------------------------------------------
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        ax = axes[idx]

        # ROC curve
        ax.plot(
            fpr,
            tpr,
            color=colors[idx],
            linewidth=2.2,
        )

        # Random-classifier reference line
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=1.0,
            alpha=0.8,
        )

        # Unified axes
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)

        ax.set_xlabel(
            "False Positive Rate",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_ylabel(
            "True Positive Rate",
            fontsize=15,
            fontweight="bold",
        )

        # Model name and AUC
        ax.set_title(
            f"{display_names[name]} (AUC = {auc:.4f})",
            fontsize=15,
            fontweight="bold",
            pad=8,
        )

        ax.tick_params(
            axis="both",
            labelsize=15,
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.6,
            alpha=0.30,
        )

        # Keep a clean publication style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(
        h_pad=2.0,
        w_pad=1.5,
    )

    fig.savefig(
        os.path.join(
            roc_curves_dir,
            "all_models_roc_subplots.pdf",
        ),
        format="pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(
            roc_curves_dir,
            "all_models_roc_subplots.tiff",
        ),
        format="tiff",
        dpi=600,
        bbox_inches="tight",
    )



    metrics_df = pd.DataFrame(metrics_list)
    model_performance_csv = os.path.join(
        BASE_OUTPUT,
        "model_performance.csv",
    )

    metrics_df.to_csv(
        model_performance_csv,
        index=False,
    )

    print("\nIndependent test-set performance:")
    print(metrics_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Model-performance visualization
    # -------------------------------------------------------------------------
    # Read the values directly from the saved result table rather than using
    # hard-coded metric values.
    model_performance_plot_dir = os.path.join(
        BASE_OUTPUT,
        "model_performance_plots",
    )
    os.makedirs(
        model_performance_plot_dir,
        exist_ok=True,
    )

    plot_model_performance_from_csv(
        csv_path=model_performance_csv,
        output_pdf=os.path.join(
            model_performance_plot_dir,
            "model_performance_subplots.pdf",
        ),
    )

    # -------------------------------------------------------------------------
    # 7. Save the ENTIRE fitted pipeline, not only the classifier
    # -------------------------------------------------------------------------
    pipeline_path = os.path.join(BASE_OUTPUT, "best_pipeline.pkl")
    joblib.dump(best_pipeline, pipeline_path)
    print(f"\nBest fitted pipeline saved to: {pipeline_path}")

    # -------------------------------------------------------------------------
    # 8. Normalized Mondrian conformal prediction
    #    Calibration set is never SMOTEd and never used for model training.
    # -------------------------------------------------------------------------
    classes, class_to_col, cal_scores_norm = build_normalized_mondrian_calibration(
        best_pipeline,
        X_cal,
        y_cal,
    )

    confidence = 0.95
    alpha = 1.0 - confidence

    test_proba_all = best_pipeline.predict_proba(X_test)
    test_entropy = compute_entropy(test_proba_all)
    test_sigma = test_entropy + 1e-6

    test_pvalues = compute_mondrian_pvalues(
        test_proba=test_proba_all,
        test_sigma=test_sigma,
        classes=classes,
        class_to_col=class_to_col,
        cal_scores_norm=cal_scores_norm,
    )

    test_sets = prediction_sets_from_pvalues(test_pvalues, classes, alpha)
    y_test_arr = np.asarray(y_test)
    covered_flags = np.asarray(
        [true_y in pred_set for true_y, pred_set in zip(y_test_arr, test_sets)]
    )

    coverage = float(np.mean(covered_flags))
    avg_set_size = float(np.mean([len(s) for s in test_sets]))

    print("\n" + "=" * 70)
    print("Normalized Mondrian conformal prediction")
    print(f"Confidence level: {confidence:.2%}")
    print(f"Empirical coverage: {coverage:.4f}")
    print(f"Average prediction-set size: {avg_set_size:.4f}")

    print("\nFirst 10 test prediction sets:")
    for i in range(min(10, len(test_sets))):
        print(f"Sample {i}: true={y_test.iloc[i]}, prediction_set={test_sets[i]}")

    # -------------------------------------------------------------------------
    # 9. Save conformal statistics
    # -------------------------------------------------------------------------
    dataset_name = "Alzheimer's disease prediction"
    rows = []

    for cls in classes:
        mask = y_test_arr == cls
        idxs = np.flatnonzero(mask)
        total = len(idxs)
        if total == 0:
            continue

        empty = 0
        single_error = 0
        single_correct = 0
        multiple = 0
        covered = 0

        for idx in idxs:
            pred_set = test_sets[idx]
            true_y = y_test_arr[idx]

            if len(pred_set) == 0:
                empty += 1
            elif len(pred_set) == 1:
                if pred_set[0] == true_y:
                    single_correct += 1
                    covered += 1
                else:
                    single_error += 1
            else:
                multiple += 1
                if true_y in pred_set:
                    covered += 1

        rows.append(
            {
                "dataset": dataset_name,
                "alpha": alpha,
                "class": cls,
                "total_in_class": total,
                "empty": empty,
                "empty_pct": empty / total * 100.0,
                "single_error": single_error,
                "single_error_pct": single_error / total * 100.0,
                "single_correct": single_correct,
                "single_correct_pct": single_correct / total * 100.0,
                "multiple": multiple,
                "multiple_pct": multiple / total * 100.0,
                "class_coverage": covered / total,
                "overall_coverage": coverage,
            }
        )

    cp_df = pd.DataFrame(rows)
    cp_df.to_csv(
        os.path.join(BASE_OUTPUT, "conformal_prediction_results.csv"),
        index=False,
        float_format="%.15f",
    )

    # -------------------------------------------------------------------------
    # 10. SHAP analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Starting SHAP analysis...")
    perform_shap_analysis(
        fitted_pipeline=best_pipeline,
        X_train_raw=X_train,
        X_test_raw=X_test,
        original_feature_names=X_train.columns.to_numpy(),
        model_name=best_name,
        output_dir="shap_plots",
    )

    # -------------------------------------------------------------------------
    # 11. Publication-style visualizations
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.5

    # -------------------------------------------------------------------------
    # Descriptive analysis of the final selected features
    # -------------------------------------------------------------------------
    # This uses the entire cohort only for post-hoc descriptive visualization.
    # It does NOT feed any full-cohort information back into model training,
    # hyperparameter tuning, model selection, conformal calibration, or SHAP.
    # -------------------------------------------------------------------------
    # Descriptive analysis of the final selected features
    # Training set only, consistent with the original manuscript.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Starting descriptive analysis of selected features on training set...")

    train_df_for_plot = X_train.copy()
    train_df_for_plot[TARGET_COL] = y_train.to_numpy()

    plot_selected_feature_descriptions(
        df=train_df_for_plot,
        selected_features=selected_features,
        target_col=TARGET_COL,
        output_dir="descriptive_analysis",
    )

    # ----- Figure 1: clearer prediction-set visualization for first 50 test samples ----
    # Main matrix:
    #   - Peach cell: the class is included in the conformal prediction set.
    #   - Blue cell: the class is not included.
    #   - Black star: the observed true class for that sample.
    #
    # Bottom annotation strip:
    #   - Singleton correct: only the true class is included.
    #   - Multiple: both classes are included (uncertain prediction).
    #   - Singleton error: only the wrong class is included.
    #   - Empty: neither class is included.
    #
    # Only the presentation of this figure is changed. The underlying prediction
    # sets, true labels, conformal calculations, and all other analyses are unchanged.
    n_show = min(50, len(test_sets))
    class_list = list(classes)

    # ---------------------------------------------------------------------
    # Build the 2 x N prediction-set membership matrix
    # ---------------------------------------------------------------------
    membership = np.zeros((len(class_list), n_show), dtype=int)

    for sample_idx in range(n_show):
        pred_set = test_sets[sample_idx]
        for row_idx, cls in enumerate(class_list):
            membership[row_idx, sample_idx] = int(cls in pred_set)

    # ---------------------------------------------------------------------
    # Determine the prediction type for each displayed sample
    # 0 = singleton correct
    # 1 = multiple-label prediction
    # 2 = singleton error
    # 3 = empty prediction set
    # ---------------------------------------------------------------------
    prediction_type_codes = np.zeros(n_show, dtype=int)

    for sample_idx in range(n_show):
        pred_set = test_sets[sample_idx]
        true_cls = y_test.iloc[sample_idx]

        if len(pred_set) == 0:
            prediction_type_codes[sample_idx] = 3
        elif len(pred_set) == 1:
            if pred_set[0] == true_cls:
                prediction_type_codes[sample_idx] = 0
            else:
                prediction_type_codes[sample_idx] = 2
        else:
            prediction_type_codes[sample_idx] = 1

    # ---------------------------------------------------------------------
    # Publication-style layout:
    # top: class-membership matrix
    # bottom: prediction-type annotation strip
    # ---------------------------------------------------------------------
    fig1 = plt.figure(figsize=(14, 10))
    gs1 = fig1.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[4,0.8],
        hspace=0.25,
    )

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax_type = fig1.add_subplot(gs1[1, 0], sharex=ax1)

    # Membership colors:
    # 0 -> not included (blue)
    # 1 -> included (peach)
    membership_cmap = ListedColormap([
        "#BFD8EB",
        "#F5C9B6",
    ])

    ax1.imshow(
        membership,
        aspect="auto",
        interpolation="none",
        cmap=membership_cmap,
        vmin=0,
        vmax=1,
    )

    # ---------------------------------------------------------------------
    # Overlay the observed true class with a black star
    # ---------------------------------------------------------------------
    for sample_idx in range(n_show):
        true_cls = y_test.iloc[sample_idx]

        if true_cls in class_list:
            true_row = class_list.index(true_cls)

            ax1.scatter(
                sample_idx,
                true_row,
                marker="*",
                s=62,
                color="#202020",
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    # ---------------------------------------------------------------------
    # Thin white boundaries between individual samples/cells
    # ---------------------------------------------------------------------
    ax1.set_xticks(
        np.arange(-0.5, n_show, 1),
        minor=True,
    )
    ax1.set_yticks(
        np.arange(-0.5, len(class_list), 1),
        minor=True,
    )
    ax1.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.65,
    )
    ax1.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Hide the x-axis labels on the main matrix.
    ax1.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    # ---------------------------------------------------------------------
    # Class labels
    # ---------------------------------------------------------------------
    class_labels = []
    for cls in class_list:
        if cls == 0:
            class_labels.append("Non-AD")
        elif cls == 1:
            class_labels.append("AD")
        else:
            class_labels.append(str(cls))

    ax1.set_yticks(np.arange(len(class_list)))
    ax1.set_yticklabels(
        class_labels,
        fontsize=15,
    )
    ax1.set_ylabel(
        "Class",
        fontsize=15,
    )

    # Remove unnecessary outer spines.
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # ---------------------------------------------------------------------
    # Prediction-type annotation strip
    # ---------------------------------------------------------------------
    # 0 = singleton correct
    # 1 = multiple
    # 2 = singleton error
    # 3 = empty
    type_colors = [
        "#BFE3C0",  # singleton correct
        "#F6D19A",  # multiple / uncertain
        "#F3B3B3",  # singleton error
        "#D7C4E8",  # empty
    ]
    type_cmap = ListedColormap(type_colors)

    ax_type.imshow(
        prediction_type_codes.reshape(1, -1),
        aspect="auto",
        interpolation="none",
        cmap=type_cmap,
        vmin=0,
        vmax=3,
    )

    # White boundaries between samples in the annotation strip.
    ax_type.set_xticks(
        np.arange(-0.5, n_show, 1),
        minor=True,
    )
    ax_type.set_yticks(
        [-0.5, 0.5],
        minor=True,
    )
    ax_type.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.65,
    )
    ax_type.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Show every fifth sample index.
    major_x = np.arange(0, n_show, 5)
    ax_type.set_xticks(major_x)
    ax_type.set_xticklabels(
        (major_x + 1).astype(int),
        fontsize=15,
    )
    ax_type.set_xlabel(
        "Test Sample Index",
        fontsize=15,
        labelpad=15,
    )

    # Label the annotation strip without crowding the figure.
    ax_type.set_yticks([0])
    ax_type.set_yticklabels(
        ["Prediction type"],
        fontsize=15,
    )

    for spine in ax_type.spines.values():
        spine.set_visible(False)

    # ---------------------------------------------------------------------
    # Legend 1: membership matrix
    # ---------------------------------------------------------------------
    membership_legend_elements = [
        Patch(
            facecolor="#F5C9B6",
            edgecolor="none",
            label="Included in prediction set",
        ),
        Patch(
            facecolor="#BFD8EB",
            edgecolor="#b0b0b0",
            label="Not included",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            label="True class",
            markerfacecolor="#202020",
            markeredgecolor="white",
            markersize=15,
        ),
    ]

    membership_legend = ax1.legend(
        handles=membership_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        facecolor="#f2f2f2",
        edgecolor="#8c8c8c",
        framealpha=1.0,
        fontsize=15,
    )
    membership_legend.get_frame().set_linewidth(0.8)

    # ---------------------------------------------------------------------
    # Legend 2: prediction types
    # ---------------------------------------------------------------------
    type_legend_elements = [
        Patch(
            facecolor=type_colors[0],
            edgecolor="none",
            label="Singleton correct",
        ),
        Patch(
            facecolor=type_colors[1],
            edgecolor="none",
            label="Multiple (uncertain)",
        ),
        Patch(
            facecolor=type_colors[2],
            edgecolor="none",
            label="Singleton error",
        ),
        Patch(
            facecolor=type_colors[3],
            edgecolor="none",
            label="Empty",
        ),
    ]

    type_legend = ax_type.legend(
        handles=type_legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.95),
        ncol=4,
        frameon=True,
        facecolor="#f2f2f2",
        edgecolor="#8c8c8c",
        framealpha=1.0,
        fontsize=15,
    )
    type_legend.get_frame().set_linewidth(0.8)

    # Keep enough room for the two legends.
    fig1.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.82,
        bottom=0.18,
    )

    # Save using the same output filename as before, so no downstream
    # manuscript references need to change.
    save_figure_pdf_tiff(
        fig1,
        os.path.join(BASE_OUTPUT, "prediction_set_matrix")
    )

    plt.close(fig1)

    # ----- Figure 2: overall and class-conditional coverage curves -----
    alpha_values = np.linspace(0.01, 0.20, 20)
    coverage_rates = []
    class_coverage_map = {cls: [] for cls in classes}
    set_sizes = []

    for a in alpha_values:
        temp_sets = prediction_sets_from_pvalues(test_pvalues, classes, a)
        temp_covered = np.asarray(
            [true_y in pred_set for true_y, pred_set in zip(y_test_arr, temp_sets)]
        )
        coverage_rates.append(float(np.mean(temp_covered)))
        set_sizes.append(float(np.mean([len(s) for s in temp_sets])))

        for cls in classes:
            mask = y_test_arr == cls
            class_coverage_map[cls].append(
                float(np.mean(temp_covered[mask])) if np.any(mask) else np.nan
            )

    theoretical_coverage = 1.0 - alpha_values

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        alpha_values,
        theoretical_coverage,
        color="#888888",
        linestyle="--",
        linewidth=1.5,
        label="Theoretical (1−α)",
    )
    ax2.plot(
        alpha_values,
        coverage_rates,
        color="#72ccff",
        linewidth=2,
        label="Overall coverage",
    )

    if 0 in class_coverage_map:
        ax2.plot(
            alpha_values,
            class_coverage_map[0],
            color="#d2f5a6",
            linewidth=2,
            linestyle="-.",
            label="Non-AD coverage",
        )
    if 1 in class_coverage_map:
        ax2.plot(
            alpha_values,
            class_coverage_map[1],
            color="#d4a4eb",
            linewidth=2,
            linestyle=":",
            label="AD coverage",
        )

    ax2.set_xlabel("Significance Level (α)", fontsize=15,fontweight="bold")
    ax2.set_ylabel("Coverage Rate", fontsize=15,fontweight="bold")
    ax2.set_xlim(0.01, 0.20)
    ax2.set_ylim(0.00, 1.00)
    coverage_legend = ax2.legend(
        loc="lower left",
        fontsize=15,
        frameon=True,
        facecolor="#f2f2f2",
        edgecolor="#8c8c8c",
        framealpha=1.0,
    )
    coverage_legend.get_frame().set_linewidth(0.8)
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    save_figure_pdf_tiff(
        fig2,
        os.path.join(BASE_OUTPUT, "coverage_curve_with_conditional")
    )
    plt.close()

    # NOTE: The previous conditional_coverage figure has been removed as requested.

    # -------------------------------------------------------------------------
    # 12. Runtime summary
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print(f"All outputs saved to: {BASE_OUTPUT}")
    print(f"Total runtime: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


if __name__ == "__main__":
    main()
