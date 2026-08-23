# =============================================================================
# Thread control
# =============================================================================
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
from scipy.stats import friedmanchisquare, wilcoxon

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =============================================================================
# Global configuration
# =============================================================================
BASE_OUTPUT = "output_ablation_rf2"
os.makedirs(BASE_OUTPUT, exist_ok=True)

DATA_PATH = r"D:\Projects\PythonProjects\PythonProject1\data\alzheimers_disease_data.csv"
TARGET_COL = "Diagnosis"

BASE_RANDOM_STATE = 42
N_REPEATS = 30

CV_FOLDS = 5
LASSO_INNER_CV_FOLDS = 5
LASSO_FIXED_ALPHA = None

N_JOBS = max(1, min(8, os.cpu_count() or 1))

PSO_PARTICLES = 20
PSO_ITERATIONS = 40

CONFIDENCE = 0.95


# =============================================================================
# Leakage-safe LASSO feature selector
# =============================================================================
class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        alpha=None,
        inner_cv_folds=5,
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
                n_jobs=1,
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

        # Never return zero features.
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
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()


# =============================================================================
# CV and pipeline
# =============================================================================
def make_cv(random_state):
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=random_state,
    )


def build_rf_pipeline(random_state):
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoFeatureSelector(
                    alpha=LASSO_FIXED_ALPHA,
                    inner_cv_folds=LASSO_INNER_CV_FOLDS,
                    random_state=random_state,
                    max_iter=10000,
                ),
            ),
            (
                "smote",
                SMOTE(random_state=random_state),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


# =============================================================================
# CPSO
# =============================================================================
class SimplePSO:

    def __init__(
        self,
        objective_func,
        bounds,
        n_particles=20,
        max_iter=40,
        w=0.7,
        c1=2.0,
        c2=2.0,
        use_chaotic_w=True,
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
        self.use_chaotic_w = use_chaotic_w
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
            Parallel(
                n_jobs=self.n_jobs,
                backend="loky",
            )(
                delayed(self.objective)(p.copy())
                for p in positions
            )
        )

    def optimize(self):
        # ---------------------------------------------------------------------
        # Chaotic initialization
        # ---------------------------------------------------------------------
        positions = np.zeros(
            (self.n_particles, self.dim),
            dtype=float,
        )

        for i in range(self.n_particles):
            x0 = (0.1 + i * 0.01) % 1.0

            if np.isclose(x0, 0.0):
                x0 = 0.001

            chaos_seq = self._logistic_sequence(
                x0,
                self.dim,
            )

            positions[i] = (
                self.lb
                + chaos_seq * (self.ub - self.lb)
            )

        velocities = self.rng.uniform(
            low=-(self.ub - self.lb) / 2.0,
            high=(self.ub - self.lb) / 2.0,
            size=(self.n_particles, self.dim),
        )

        # ---------------------------------------------------------------------
        # Chaotic r1 / r2 states
        # ---------------------------------------------------------------------
        mu = 4.0

        chaos_r1 = np.zeros_like(positions)
        chaos_r2 = np.zeros_like(positions)

        for i in range(self.n_particles):
            for j in range(self.dim):
                init1 = (
                    0.1
                    + i * 0.01
                    + j * 0.001
                ) % 1.0

                init2 = (
                    0.2
                    + i * 0.01
                    + j * 0.001
                ) % 1.0

                chaos_r1[i, j] = (
                    0.001
                    if np.isclose(init1, 0.0)
                    else init1
                )
                chaos_r2[i, j] = (
                    0.001
                    if np.isclose(init2, 0.0)
                    else init2
                )

        # ---------------------------------------------------------------------
        # Chaotic inertia-weight states
        # ---------------------------------------------------------------------
        if self.use_chaotic_w:
            chaos_w = self.rng.uniform(
                0.001,
                0.999,
                size=(self.n_particles, 1),
            )
        else:
            chaos_w = None

        personal_best = positions.copy()
        personal_best_score = (
            self._evaluate_positions(positions)
        )

        best_idx = int(
            np.argmax(personal_best_score)
        )

        global_best = (
            personal_best[best_idx].copy()
        )
        global_best_score = float(
            personal_best_score[best_idx]
        )

        # ---------------------------------------------------------------------
        # Main loop
        # ---------------------------------------------------------------------
        for iteration in range(self.max_iter):
            r1 = chaos_r1.copy()
            r2 = chaos_r2.copy()

            chaos_r1 = (
                mu
                * chaos_r1
                * (1.0 - chaos_r1)
            )
            chaos_r2 = (
                mu
                * chaos_r2
                * (1.0 - chaos_r2)
            )

            if self.use_chaotic_w:
                chaos_w = (
                    mu
                    * chaos_w
                    * (1.0 - chaos_w)
                )

                w_current = (
                    self.w
                    * np.broadcast_to(
                        chaos_w,
                        velocities.shape,
                    )
                )
            else:
                w_current = self.w

            velocities = (
                w_current * velocities
                + self.c1
                * r1
                * (personal_best - positions)
                + self.c2
                * r2
                * (global_best - positions)
            )

            positions = np.clip(
                positions + velocities,
                self.lb,
                self.ub,
            )

            scores = self._evaluate_positions(
                positions
            )

            improve = (
                scores
                > personal_best_score
            )

            personal_best[improve] = (
                positions[improve]
            )
            personal_best_score[improve] = (
                scores[improve]
            )

            iter_best_idx = int(
                np.argmax(scores)
            )
            iter_best_score = float(
                scores[iter_best_idx]
            )

            if (
                iter_best_score
                > global_best_score
            ):
                global_best = (
                    positions[
                        iter_best_idx
                    ].copy()
                )
                global_best_score = (
                    iter_best_score
                )

            print(
                f"      CPSO iteration "
                f"{iteration + 1:02d}/"
                f"{self.max_iter}: "
                f"best CV AUC="
                f"{global_best_score:.4f}"
            )

        return (
            global_best,
            global_best_score,
        )


# =============================================================================
# Mixed Grid/CPSO parameter-space helpers
# =============================================================================
def _is_pure_numeric(values):
    """
    True only if every option is numeric and none is None/bool.
    """
    if len(values) == 0:
        return False

    for value in values:
        if value is None:
            return False

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return False

        if not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            return False

    return True


def _all_integer(values):
    return all(
        isinstance(
            v,
            (int, np.integer),
        )
        and not isinstance(
            v,
            (bool, np.bool_),
        )
        for v in values
    )


def build_pso_specs(
    param_grid,
    center_params=None,
    local=False,
):
    specs = []
    bounds = []

    for key, values in param_grid.items():
        values = list(values)

        if _is_pure_numeric(values):
            low = float(min(values))
            high = float(max(values))

            if local and center_params is not None:
                center = float(
                    center_params[key]
                )
                span = high - low

                low = max(
                    low,
                    center - 0.2 * span,
                )
                high = min(
                    high,
                    center + 0.2 * span,
                )

            specs.append(
                {
                    "key": key,
                    "kind": "numeric",
                    "integer": _all_integer(values),
                    "values": values,
                }
            )
            bounds.append(
                (low, high)
            )

        else:
            # Search the SAME allowed discrete choices using encoded indices.
            if local and center_params is not None:
                center_value = (
                    center_params[key]
                )
                center_idx = values.index(
                    center_value
                )

                low = max(
                    0,
                    center_idx - 1,
                )
                high = min(
                    len(values) - 1,
                    center_idx + 1,
                )
            else:
                low = 0
                high = len(values) - 1

            specs.append(
                {
                    "key": key,
                    "kind": "categorical",
                    "integer": True,
                    "values": values,
                }
            )
            bounds.append(
                (float(low), float(high))
            )

    return specs, bounds


def decode_pso_vector(
    vec,
    specs,
):
    params = {}

    for i, spec in enumerate(specs):
        value = float(vec[i])

        if (
            spec["kind"]
            == "numeric"
        ):
            if spec["integer"]:
                value = int(
                    round(value)
                )

                # RF integer hyperparameters
                # must stay positive.
                value = max(
                    1,
                    value,
                )

            params[
                spec["key"]
            ] = value

        else:
            idx = int(
                round(value)
            )
            idx = int(
                np.clip(
                    idx,
                    0,
                    len(
                        spec["values"]
                    ) - 1,
                )
            )

            params[
                spec["key"]
            ] = (
                spec["values"][idx]
            )

    return params


# =============================================================================
# Optimization methods
# =============================================================================
def get_grid_params(
    base_pipeline,
    X,
    y,
    param_grid,
    random_state,
):
    # Measure only the hyperparameter-search stage.
    start_time = time.perf_counter()

    grid = GridSearchCV(
        estimator=clone(base_pipeline),
        param_grid=param_grid,
        cv=make_cv(random_state),
        scoring="roc_auc",
        n_jobs=N_JOBS,
        refit=False,
        error_score="raise",
    )

    grid.fit(X, y)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"    Grid best CV AUC="
        f"{grid.best_score_:.4f}; "
        f"tuning time="
        f"{elapsed:.2f} s"
    )

    return (
        grid.best_params_.copy(),
        float(grid.best_score_),
        float(elapsed),
    )



def get_random_params(
    base_pipeline,
    X,
    y,
    param_grid,
    random_state,
):
    """
    Random Search baseline with the same CV and scoring settings.
    """
    start_time = time.perf_counter()

    random_search = RandomizedSearchCV(
        estimator=clone(base_pipeline),
        param_distributions=param_grid,
        n_iter=10,
        cv=make_cv(random_state),
        scoring="roc_auc",
        n_jobs=N_JOBS,
        refit=False,
        random_state=random_state,
        error_score="raise",
    )

    random_search.fit(X, y)

    elapsed = time.perf_counter() - start_time

    print(
        f"    Random Search best CV AUC="
        f"{random_search.best_score_:.4f}; "
        f"tuning time={elapsed:.2f} s"
    )

    return (
        random_search.best_params_.copy(),
        float(random_search.best_score_),
        float(elapsed),
    )


def get_pso_only_params(
    base_pipeline,
    X,
    y,
    param_grid,
    random_state,
):
    # Measure only the hyperparameter-search stage.
    start_time = time.perf_counter()

    specs, bounds = build_pso_specs(
        param_grid,
        center_params=None,
        local=False,
    )

    def objective(vec):
        params = decode_pso_vector(
            vec,
            specs,
        )

        candidate = clone(
            base_pipeline
        )
        candidate.set_params(
            **params
        )

        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=make_cv(
                random_state
            ),
            scoring="roc_auc",
            n_jobs=1,
            error_score="raise",
        )

        return float(
            np.mean(scores)
        )

    pso = SimplePSO(
        objective_func=objective,
        bounds=bounds,
        n_particles=PSO_PARTICLES,
        max_iter=PSO_ITERATIONS,
        use_chaotic_w=True,
        n_jobs=N_JOBS,
        random_state=random_state,
    )

    best_vec, best_score = (
        pso.optimize()
    )

    best_params = decode_pso_vector(
        best_vec,
        specs,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"    CPSO-only best CV AUC="
        f"{best_score:.4f}; "
        f"tuning time="
        f"{elapsed:.2f} s"
    )

    return (
        best_params,
        float(best_score),
        float(elapsed),
    )


def get_grid_pso_params(
    base_pipeline,
    X,
    y,
    param_grid,
    random_state,
):
    # Measure the complete hybrid tuning procedure:
    # Grid Search stage + local CPSO refinement stage.
    total_start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # Stage 1: Grid
    # -------------------------------------------------------------------------
    (
        grid_params,
        grid_score,
        grid_time,
    ) = get_grid_params(
        base_pipeline,
        X,
        y,
        param_grid,
        random_state,
    )

    # -------------------------------------------------------------------------
    # Stage 2: local CPSO refinement
    # -------------------------------------------------------------------------
    cpso_start_time = time.perf_counter()

    specs, bounds = build_pso_specs(
        param_grid,
        center_params=grid_params,
        local=True,
    )

    def objective(vec):
        local_params = decode_pso_vector(
            vec,
            specs,
        )

        # CPSO is refining around the grid solution.
        candidate_params = (
            grid_params.copy()
        )
        candidate_params.update(
            local_params
        )

        candidate = clone(
            base_pipeline
        )
        candidate.set_params(
            **candidate_params
        )

        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=make_cv(
                random_state
            ),
            scoring="roc_auc",
            n_jobs=1,
            error_score="raise",
        )

        return float(
            np.mean(scores)
        )

    pso = SimplePSO(
        objective_func=objective,
        bounds=bounds,
        n_particles=PSO_PARTICLES,
        max_iter=PSO_ITERATIONS,
        use_chaotic_w=True,
        n_jobs=N_JOBS,
        random_state=random_state,
    )

    best_vec, cpso_score = (
        pso.optimize()
    )

    cpso_time = (
        time.perf_counter()
        - cpso_start_time
    )

    cpso_params = (
        grid_params.copy()
    )
    cpso_params.update(
        decode_pso_vector(
            best_vec,
            specs,
        )
    )

    # Hybrid method should never discard a better grid solution.
    if cpso_score > grid_score:
        final_params = cpso_params
        final_score = float(
            cpso_score
        )
        source = "CPSO_refinement"
    else:
        final_params = grid_params
        final_score = float(
            grid_score
        )
        source = "Grid_retained"

    total_time = (
        time.perf_counter()
        - total_start_time
    )

    print(
        f"    Grid+CPSO selected CV AUC="
        f"{final_score:.4f} "
        f"({source})"
    )
    print(
        f"    Grid stage time="
        f"{grid_time:.2f} s; "
        f"CPSO refinement time="
        f"{cpso_time:.2f} s; "
        f"total tuning time="
        f"{total_time:.2f} s"
    )

    return (
        final_params,
        final_score,
        float(total_time),
        float(grid_time),
        float(cpso_time),
    )


# =============================================================================
# Normalized Mondrian conformal prediction
# =============================================================================
def compute_entropy(
    proba,
    epsilon=1e-12,
):
    p = np.clip(
        np.asarray(
            proba,
            dtype=float,
        ),
        epsilon,
        1.0 - epsilon,
    )

    return -np.sum(
        p * np.log(p),
        axis=1,
    )


def evaluate_conformal(
    fitted_pipeline,
    X_cal,
    y_cal,
    X_test,
    y_test,
    confidence=0.95,
):
    """
    Marginal proportions, aligned with the main experiment.
    """
    cal_proba = (
        fitted_pipeline.predict_proba(
            X_cal
        )
    )

    classifier = (
        fitted_pipeline.named_steps[
            "classifier"
        ]
    )

    classes = np.asarray(
        classifier.classes_
    )

    class_to_col = {
        cls: idx
        for idx, cls
        in enumerate(classes)
    }

    cal_entropy = compute_entropy(
        cal_proba
    )
    cal_sigma = (
        cal_entropy + 1e-6
    )

    y_cal_arr = np.asarray(
        y_cal
    )

    cal_scores_norm = {}

    for cls in classes:
        col = class_to_col[cls]
        mask = (
            y_cal_arr == cls
        )

        raw_scores = (
            1.0
            - cal_proba[
                mask,
                col,
            ]
        )

        cal_scores_norm[cls] = (
            np.sort(
                raw_scores
                / cal_sigma[mask]
            )
        )

    alpha = (
        1.0 - confidence
    )

    test_proba = (
        fitted_pipeline.predict_proba(
            X_test
        )
    )

    test_entropy = compute_entropy(
        test_proba
    )
    test_sigma = (
        test_entropy + 1e-6
    )

    test_sets = []

    for i in range(
        len(X_test)
    ):
        pred_set = []

        for cls in classes:
            col = (
                class_to_col[cls]
            )

            raw_score = (
                1.0
                - test_proba[
                    i,
                    col,
                ]
            )

            norm_score = (
                raw_score
                / test_sigma[i]
            )

            cal_scores = (
                cal_scores_norm[cls]
            )

            p_value = (
                np.sum(
                    cal_scores
                    >= norm_score
                )
                + 1.0
            ) / (
                len(cal_scores)
                + 1.0
            )

            if p_value > alpha:
                pred_set.append(
                    cls
                )

        test_sets.append(
            pred_set
        )

    y_test_arr = np.asarray(
        y_test
    )

    covered = np.asarray(
        [
            true_y in pred_set
            for true_y, pred_set
            in zip(
                y_test_arr,
                test_sets,
            )
        ]
    )

    coverage = float(
        np.mean(covered)
    )

    avg_set_size = float(
        np.mean(
            [
                len(s)
                for s in test_sets
            ]
        )
    )

    class_coverage = {}

    for cls in classes:
        mask = (
            y_test_arr == cls
        )

        class_coverage[cls] = (
            float(
                np.mean(
                    covered[mask]
                )
            )
            if np.any(mask)
            else np.nan
        )

    total = len(
        y_test_arr
    )

    single_correct_cnt = 0
    single_error_cnt = 0
    multiple_cnt = 0
    empty_cnt = 0

    for i, pred_set in enumerate(
        test_sets
    ):
        if len(pred_set) == 0:
            empty_cnt += 1

        elif len(pred_set) == 1:
            if (
                pred_set[0]
                == y_test_arr[i]
            ):
                single_correct_cnt += 1
            else:
                single_error_cnt += 1

        else:
            multiple_cnt += 1

    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "coverage_class0": (
            class_coverage.get(
                0,
                np.nan,
            )
        ),
        "coverage_class1": (
            class_coverage.get(
                1,
                np.nan,
            )
        ),
        "alpha": alpha,
        "single_correct": (
            single_correct_cnt
            / total
        ),
        "single_error": (
            single_error_cnt
            / total
        ),
        "multiple_pct": (
            multiple_cnt
            / total
        ),
        "empty_pct": (
            empty_cnt
            / total
        ),
    }


# =============================================================================
# Data split
# =============================================================================
def load_and_split(
    random_state,
):
    df = pd.read_csv(
        DATA_PATH
    )

    X = df.drop(
        columns=[TARGET_COL]
    )
    y = df[TARGET_COL]

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.4,
            random_state=random_state,
            stratify=y,
        )
    )

    X_cal, X_test, y_cal, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=random_state,
            stratify=y_temp,
        )
    )

    return (
        X_train.reset_index(
            drop=True
        ),
        y_train.reset_index(
            drop=True
        ),
        X_cal.reset_index(
            drop=True
        ),
        y_cal.reset_index(
            drop=True
        ),
        X_test.reset_index(
            drop=True
        ),
        y_test.reset_index(
            drop=True
        ),
    )



# =============================================================================
# Statistical significance testing
# =============================================================================
def holm_adjust_pvalues(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for i, p in enumerate(sorted_p):
        adjusted = (m - i) * p
        running_max = max(running_max, adjusted)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    return adjusted


def run_statistical_tests(result_df, output_dir):
    required_methods = [
        "Grid_Search",
        "Random_Search",
        "CPSO_only",
        "Grid_CPSO",
    ]

    # -------------------------------------------------------------------------
    # Pivot to paired wide format: one row per seed/repeat.
    # -------------------------------------------------------------------------
    paired = result_df.pivot_table(
        index=["repeat", "seed"],
        columns="method",
        values="gmean",
        aggfunc="first",
    )

    missing_methods = [
        method
        for method in required_methods
        if method not in paired.columns
    ]
    if missing_methods:
        raise ValueError(
            "Statistical testing cannot proceed because methods are missing: "
            f"{missing_methods}"
        )

    paired = paired[required_methods].dropna().copy()

    if len(paired) < 2:
        raise ValueError(
            "Insufficient paired runs for statistical testing."
        )

    # Save the paired raw G-mean data used for testing.
    paired.reset_index().to_csv(
        os.path.join(
            output_dir,
            "gmean_paired_runs_for_statistics.csv",
        ),
        index=False,
    )

    grid = paired["Grid_Search"].to_numpy(dtype=float)
    random_search = paired["Random_Search"].to_numpy(dtype=float)
    cpso = paired["CPSO_only"].to_numpy(dtype=float)
    hybrid = paired["Grid_CPSO"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Overall Friedman test.
    # -------------------------------------------------------------------------
    friedman_stat, friedman_p = friedmanchisquare(
        grid,
        random_search,
        cpso,
        hybrid,
    )

    friedman_df = pd.DataFrame(
        [
            {
                "metric": "G-mean",
                "test": "Friedman",
                "n_paired_runs": len(paired),
                "methods": "Grid_Search | CPSO_only | Grid_CPSO",
                "statistic": float(friedman_stat),
                "p_value": float(friedman_p),
                "significant_at_0.05": bool(friedman_p < 0.05),
            }
        ]
    )

    friedman_df.to_csv(
        os.path.join(
            output_dir,
            "friedman_test_gmean.csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Pairwise Wilcoxon signed-rank tests.
    # -------------------------------------------------------------------------
    comparisons = [
        ("Grid_CPSO", "Grid_Search", hybrid, grid),
        ("Grid_CPSO", "Random_Search", hybrid, random_search),
        ("Grid_CPSO", "CPSO_only", hybrid, cpso),
        ("Grid_Search", "Random_Search", grid, random_search),
        ("Grid_Search", "CPSO_only", grid, cpso),
        ("Random_Search", "CPSO_only", random_search, cpso),
    ]

    pairwise_rows = []

    for method_a, method_b, values_a, values_b in comparisons:
        differences = values_a - values_b

        # scipy wilcoxon can fail when all paired differences are exactly zero.
        if np.allclose(differences, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )

        pairwise_rows.append(
            {
                "metric": "G-mean",
                "test": "Wilcoxon signed-rank",
                "method_A": method_a,
                "method_B": method_b,
                "n_paired_runs": len(paired),
                "mean_method_A": float(np.mean(values_a)),
                "mean_method_B": float(np.mean(values_b)),
                "mean_difference_A_minus_B": float(
                    np.mean(differences)
                ),
                "median_difference_A_minus_B": float(
                    np.median(differences)
                ),
                "n_A_greater_B": int(
                    np.sum(differences > 0)
                ),
                "n_A_equal_B": int(
                    np.sum(np.isclose(differences, 0.0))
                ),
                "n_A_less_B": int(
                    np.sum(differences < 0)
                ),
                "statistic": float(statistic),
                "p_value_raw": float(p_value),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)

    pairwise_df["p_value_holm"] = holm_adjust_pvalues(
        pairwise_df["p_value_raw"].to_numpy()
    )

    pairwise_df["significant_raw_0.05"] = (
        pairwise_df["p_value_raw"] < 0.05
    )
    pairwise_df["significant_holm_0.05"] = (
        pairwise_df["p_value_holm"] < 0.05
    )

    pairwise_df.to_csv(
        os.path.join(
            output_dir,
            "wilcoxon_pairwise_gmean_holm.csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Combined statistical summary table.
    # -------------------------------------------------------------------------
    combined_rows = [
        {
            "analysis_level": "overall",
            "metric": "G-mean",
            "test": "Friedman",
            "comparison": (
                "Grid_Search vs CPSO_only vs Grid_CPSO"
            ),
            "n_paired_runs": len(paired),
            "statistic": float(friedman_stat),
            "p_value_raw": float(friedman_p),
            "p_value_adjusted": np.nan,
            "significant_at_0.05": bool(
                friedman_p < 0.05
            ),
        }
    ]

    for _, row in pairwise_df.iterrows():
        combined_rows.append(
            {
                "analysis_level": "pairwise",
                "metric": "G-mean",
                "test": "Wilcoxon signed-rank",
                "comparison": (
                    f"{row['method_A']} vs {row['method_B']}"
                ),
                "n_paired_runs": int(
                    row["n_paired_runs"]
                ),
                "statistic": float(
                    row["statistic"]
                ),
                "p_value_raw": float(
                    row["p_value_raw"]
                ),
                "p_value_adjusted": float(
                    row["p_value_holm"]
                ),
                "significant_at_0.05": bool(
                    row["significant_holm_0.05"]
                ),
            }
        )

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        os.path.join(
            output_dir,
            "statistical_significance_summary.csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console output.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Statistical significance testing for G-mean")
    print("=" * 78)

    print(
        f"Friedman: statistic={friedman_stat:.6f}, "
        f"p={friedman_p:.6g}"
    )

    print("\nPairwise Wilcoxon tests with Holm correction:")
    print(
        pairwise_df[
            [
                "method_A",
                "method_B",
                "mean_difference_A_minus_B",
                "p_value_raw",
                "p_value_holm",
                "significant_holm_0.05",
            ]
        ].to_string(
            index=False
        )
    )

    return (
        friedman_df,
        pairwise_df,
        combined_df,
    )


# =============================================================================
# Runtime statistical analysis
# =============================================================================
def run_runtime_tests(result_df, output_dir):
    """
    Compare the wall-clock hyperparameter-tuning time of the three optimization
    strategies across the same 30 paired repeats.

    The primary runtime variable is `tuning_seconds`, which includes only the
    optimization/search stage. Final model fitting and conformal prediction are
    not included in this comparison.
    """
    required_methods = [
        "Grid_Search",
        "Random_Search",
        "CPSO_only",
        "Grid_CPSO",
    ]

    paired = result_df.pivot_table(
        index=["repeat", "seed"],
        columns="method",
        values="tuning_seconds",
        aggfunc="first",
    )

    missing_methods = [
        method
        for method in required_methods
        if method not in paired.columns
    ]
    if missing_methods:
        raise ValueError(
            "Runtime testing cannot proceed because methods are missing: "
            f"{missing_methods}"
        )

    paired = (
        paired[required_methods]
        .dropna()
        .copy()
    )

    if len(paired) < 2:
        raise ValueError(
            "Insufficient paired runs for runtime testing."
        )

    paired.reset_index().to_csv(
        os.path.join(
            output_dir,
            "runtime_paired_runs_for_statistics.csv",
        ),
        index=False,
    )

    grid = paired["Grid_Search"].to_numpy(dtype=float)
    random_search = paired["Random_Search"].to_numpy(dtype=float)
    cpso = paired["CPSO_only"].to_numpy(dtype=float)
    hybrid = paired["Grid_CPSO"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # Overall Friedman test
    # -------------------------------------------------------------------------
    friedman_stat, friedman_p = friedmanchisquare(
        grid,
        random_search,
        cpso,
        hybrid,
    )

    friedman_df = pd.DataFrame(
        [
            {
                "metric": "Tuning runtime (seconds)",
                "test": "Friedman",
                "n_paired_runs": len(paired),
                "methods": "Grid_Search | CPSO_only | Grid_CPSO",
                "statistic": float(friedman_stat),
                "p_value": float(friedman_p),
                "significant_at_0.05": bool(friedman_p < 0.05),
            }
        ]
    )

    friedman_df.to_csv(
        os.path.join(
            output_dir,
            "friedman_test_runtime.csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Pairwise Wilcoxon signed-rank tests
    # -------------------------------------------------------------------------
    comparisons = [
        ("Grid_CPSO", "Grid_Search", hybrid, grid),
        ("Grid_CPSO", "Random_Search", hybrid, random_search),
        ("Grid_CPSO", "CPSO_only", hybrid, cpso),
        ("Grid_Search", "Random_Search", grid, random_search),
        ("Grid_Search", "CPSO_only", grid, cpso),
        ("Random_Search", "CPSO_only", random_search, cpso),
    ]

    pairwise_rows = []

    for method_a, method_b, values_a, values_b in comparisons:
        differences = (
            values_a
            - values_b
        )

        if np.allclose(
            differences,
            0.0,
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )

        pairwise_rows.append(
            {
                "metric": "Tuning runtime (seconds)",
                "test": "Wilcoxon signed-rank",
                "method_A": method_a,
                "method_B": method_b,
                "n_paired_runs": len(paired),
                "mean_method_A_seconds": float(np.mean(values_a)),
                "mean_method_B_seconds": float(np.mean(values_b)),
                "mean_difference_A_minus_B_seconds": float(
                    np.mean(differences)
                ),
                "median_difference_A_minus_B_seconds": float(
                    np.median(differences)
                ),
                "n_A_greater_B": int(
                    np.sum(differences > 0)
                ),
                "n_A_equal_B": int(
                    np.sum(np.isclose(differences, 0.0))
                ),
                "n_A_less_B": int(
                    np.sum(differences < 0)
                ),
                "statistic": float(statistic),
                "p_value_raw": float(p_value),
            }
        )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    pairwise_df["p_value_holm"] = holm_adjust_pvalues(
        pairwise_df[
            "p_value_raw"
        ].to_numpy()
    )

    pairwise_df["significant_raw_0.05"] = (
        pairwise_df[
            "p_value_raw"
        ] < 0.05
    )

    pairwise_df["significant_holm_0.05"] = (
        pairwise_df[
            "p_value_holm"
        ] < 0.05
    )

    pairwise_df.to_csv(
        os.path.join(
            output_dir,
            "wilcoxon_pairwise_runtime_holm.csv",
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Combined runtime statistical summary
    # -------------------------------------------------------------------------
    combined_rows = [
        {
            "analysis_level": "overall",
            "metric": "Tuning runtime (seconds)",
            "test": "Friedman",
            "comparison": (
                "Grid_Search vs CPSO_only vs Grid_CPSO"
            ),
            "n_paired_runs": len(paired),
            "statistic": float(friedman_stat),
            "p_value_raw": float(friedman_p),
            "p_value_adjusted": np.nan,
            "significant_at_0.05": bool(
                friedman_p < 0.05
            ),
        }
    ]

    for _, row in pairwise_df.iterrows():
        combined_rows.append(
            {
                "analysis_level": "pairwise",
                "metric": "Tuning runtime (seconds)",
                "test": "Wilcoxon signed-rank",
                "comparison": (
                    f"{row['method_A']} vs {row['method_B']}"
                ),
                "n_paired_runs": int(
                    row["n_paired_runs"]
                ),
                "statistic": float(
                    row["statistic"]
                ),
                "p_value_raw": float(
                    row["p_value_raw"]
                ),
                "p_value_adjusted": float(
                    row["p_value_holm"]
                ),
                "significant_at_0.05": bool(
                    row["significant_holm_0.05"]
                ),
            }
        )

    combined_df = pd.DataFrame(
        combined_rows
    )

    combined_df.to_csv(
        os.path.join(
            output_dir,
            "runtime_statistical_significance_summary.csv",
        ),
        index=False,
    )

    print("\n" + "=" * 78)
    print("Statistical significance testing for tuning runtime")
    print("=" * 78)
    print(
        f"Friedman: statistic={friedman_stat:.6f}, "
        f"p={friedman_p:.6g}"
    )

    print(
        "\nPairwise Wilcoxon runtime tests with Holm correction:"
    )
    print(
        pairwise_df[
            [
                "method_A",
                "method_B",
                "mean_difference_A_minus_B_seconds",
                "p_value_raw",
                "p_value_holm",
                "significant_holm_0.05",
            ]
        ].to_string(
            index=False
        )
    )

    return (
        friedman_df,
        pairwise_df,
        combined_df,
    )


# =============================================================================
# Main ablation
# =============================================================================
def run_ablation():
    start_time = (
        time.perf_counter()
    )

    param_grid = {
        "classifier__n_estimators": [
            50,
            100,
            200,
        ],
        "classifier__max_depth": [
            3,
            5,
            10,
            None,
        ],
    }

    method_names = [
        "Grid_Search",
        "Random_Search",
        "CPSO_only",
        "Grid_CPSO",
    ]

    results = []

    # -------------------------------------------------------------------------
    # Loop by seed FIRST so all methods use exactly the same split for a repeat.
    # -------------------------------------------------------------------------
    for repeat_idx in range(
        N_REPEATS
    ):
        seed = (
            BASE_RANDOM_STATE
            + repeat_idx
        )

        print(
            "\n"
            + "=" * 78
        )
        print(
            f"Repeat "
            f"{repeat_idx + 1}/"
            f"{N_REPEATS}, "
            f"seed={seed}"
        )
        print(
            "=" * 78
        )

        (
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_test,
            y_test,
        ) = load_and_split(
            random_state=seed
        )

        # ---------------------------------------------------------------------
        # Same data split for Grid / CPSO / Grid+CPSO.
        # ---------------------------------------------------------------------
        for method_name in method_names:
            print(
                f"\n  ----- "
                f"{method_name} "
                f"-----"
            )

            base_pipeline = (
                build_rf_pipeline(
                    random_state=seed
                )
            )

            # -----------------------------------------------------------------
            # Hyperparameter tuning runtime.
            # Only the optimization/search stage is timed here.
            # -----------------------------------------------------------------
            if (
                method_name
                == "Grid_Search"
            ):
                (
                    params,
                    tuning_auc,
                    tuning_seconds,
                ) = get_grid_params(
                    base_pipeline,
                    X_train,
                    y_train,
                    param_grid,
                    seed,
                )

                grid_stage_seconds = (
                    tuning_seconds
                )
                cpso_stage_seconds = (
                    0.0
                )

            elif (
                method_name
                == "Random_Search"
            ):
                (
                    params,
                    tuning_auc,
                    tuning_seconds,
                ) = get_random_params(
                    base_pipeline,
                    X_train,
                    y_train,
                    param_grid,
                    seed,
                )

                grid_stage_seconds = 0.0
                cpso_stage_seconds = 0.0

            elif (
                method_name
                == "CPSO_only"
            ):
                (
                    params,
                    tuning_auc,
                    tuning_seconds,
                ) = get_pso_only_params(
                    base_pipeline,
                    X_train,
                    y_train,
                    param_grid,
                    seed,
                )

                grid_stage_seconds = (
                    0.0
                )
                cpso_stage_seconds = (
                    tuning_seconds
                )

            else:
                (
                    params,
                    tuning_auc,
                    tuning_seconds,
                    grid_stage_seconds,
                    cpso_stage_seconds,
                ) = get_grid_pso_params(
                    base_pipeline,
                    X_train,
                    y_train,
                    param_grid,
                    seed,
                )

            # -----------------------------------------------------------------
            # Final fit on the entire 60% training set only.
            # This is timed separately and is NOT included in the primary
            # optimization-runtime comparison.
            # -----------------------------------------------------------------
            final_pipeline = clone(
                base_pipeline
            )
            final_pipeline.set_params(
                **params
            )

            fit_start = (
                time.perf_counter()
            )

            final_pipeline.fit(
                X_train,
                y_train,
            )

            final_fit_seconds = (
                time.perf_counter()
                - fit_start
            )

            # -----------------------------------------------------------------
            # Independent test performance.
            # -----------------------------------------------------------------
            y_pred = (
                final_pipeline.predict(
                    X_test
                )
            )

            tn, fp, fn, tp = (
                confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                ).ravel()
            )

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0.0
            )

            sensitivity = (
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0,
                )
            )

            gmean = float(
                np.sqrt(
                    sensitivity
                    * specificity
                )
            )

            # -----------------------------------------------------------------
            # Conformal calibration uses ONLY the 20% calibration set.
            # -----------------------------------------------------------------
            conformal_metrics = (
                evaluate_conformal(
                    final_pipeline,
                    X_cal,
                    y_cal,
                    X_test,
                    y_test,
                    confidence=CONFIDENCE,
                )
            )

            selector = (
                final_pipeline.named_steps[
                    "lasso"
                ]
            )

            selected_mask = (
                selector.get_support()
            )

            selected_features = (
                X_train.columns[
                    selected_mask
                ].tolist()
            )

            result_entry = {
                "method": method_name,
                "repeat": (
                    repeat_idx + 1
                ),
                "seed": seed,
                "tuning_cv_auc": (
                    tuning_auc
                ),
                # Primary runtime comparison:
                # complete hyperparameter-search time for this strategy.
                "tuning_seconds": (
                    tuning_seconds
                ),
                # Stage-specific times are useful for decomposing Grid+CPSO.
                "grid_stage_seconds": (
                    grid_stage_seconds
                ),
                "cpso_stage_seconds": (
                    cpso_stage_seconds
                ),
                # Final fitting time is stored separately.
                "final_fit_seconds": (
                    final_fit_seconds
                ),
                "gmean": gmean,
                "sensitivity": (
                    sensitivity
                ),
                "specificity": (
                    specificity
                ),
                "lasso_alpha": (
                    selector.alpha_
                ),
                "n_selected_features": (
                    len(
                        selected_features
                    )
                ),
                "selected_features": (
                    "|".join(
                        selected_features
                    )
                ),
                "best_params": (
                    repr(params)
                ),
            }

            for key, value in (
                conformal_metrics.items()
            ):
                result_entry[
                    f"cp_{key}"
                ] = value

            results.append(
                result_entry
            )

            print(
                f"    Test G-mean="
                f"{gmean:.4f}"
            )
            print(
                f"    Tuning runtime="
                f"{tuning_seconds:.2f} s"
            )
            print(
                f"    Final fit runtime="
                f"{final_fit_seconds:.2f} s"
            )
            print(
                f"    CP coverage="
                f"{conformal_metrics['coverage']:.4f}, "
                f"avg set size="
                f"{conformal_metrics['avg_set_size']:.4f}"
            )

    # =========================================================================
    # Save repeat-level results
    # =========================================================================
    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "ablation_results_by_repeat.csv",
        ),
        index=False,
    )

    # =========================================================================
    # Performance + runtime summary
    # =========================================================================
    summary_rows = []

    cp_cols = [
        col
        for col in result_df.columns
        if col.startswith("cp_")
    ]

    for method in method_names:
        sub = result_df[
            result_df["method"]
            == method
        ]

        row = {
            "method": method,
            "gmean_mean": (
                sub["gmean"].mean()
            ),
            "gmean_min": (
                sub["gmean"].min()
            ),
            "gmean_max": (
                sub["gmean"].max()
            ),
            "gmean_sd": (
                sub["gmean"].std(
                    ddof=1
                )
            ),
            "tuning_cv_auc_mean": (
                sub[
                    "tuning_cv_auc"
                ].mean()
            ),
            "tuning_seconds_mean": (
                sub[
                    "tuning_seconds"
                ].mean()
            ),
            "tuning_seconds_median": (
                sub[
                    "tuning_seconds"
                ].median()
            ),
            "tuning_seconds_min": (
                sub[
                    "tuning_seconds"
                ].min()
            ),
            "tuning_seconds_max": (
                sub[
                    "tuning_seconds"
                ].max()
            ),
            "tuning_seconds_sd": (
                sub[
                    "tuning_seconds"
                ].std(
                    ddof=1
                )
            ),
            "final_fit_seconds_mean": (
                sub[
                    "final_fit_seconds"
                ].mean()
            ),
        }

        for col in cp_cols:
            row[
                col.replace(
                    "cp_",
                    "cp_mean_",
                )
            ] = sub[col].mean()

        summary_rows.append(
            row
        )

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .round(4)
    )

    summary_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "ablation_summary.csv",
        ),
        index=False,
    )

    # =========================================================================
    # Dedicated runtime summary
    # =========================================================================
    runtime_summary_rows = []

    grid_mean_time = float(
        result_df.loc[
            result_df["method"]
            == "Grid_Search",
            "tuning_seconds",
        ].mean()
    )

    for method in method_names:
        sub = result_df[
            result_df["method"]
            == method
        ]

        mean_time = float(
            sub[
                "tuning_seconds"
            ].mean()
        )

        runtime_summary_rows.append(
            {
                "method": method,
                "n_runs": len(sub),
                "tuning_seconds_mean": mean_time,
                "tuning_seconds_median": float(
                    sub["tuning_seconds"].median()
                ),
                "tuning_seconds_sd": float(
                    sub["tuning_seconds"].std(ddof=1)
                ),
                "tuning_seconds_min": float(
                    sub["tuning_seconds"].min()
                ),
                "tuning_seconds_max": float(
                    sub["tuning_seconds"].max()
                ),
                "relative_runtime_vs_grid": (
                    mean_time / grid_mean_time
                    if grid_mean_time > 0
                    else np.nan
                ),
                "grid_stage_seconds_mean": float(
                    sub["grid_stage_seconds"].mean()
                ),
                "cpso_stage_seconds_mean": float(
                    sub["cpso_stage_seconds"].mean()
                ),
                "final_fit_seconds_mean": float(
                    sub["final_fit_seconds"].mean()
                ),
            }
        )

    runtime_summary_df = (
        pd.DataFrame(
            runtime_summary_rows
        )
        .round(4)
    )

    runtime_summary_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "runtime_summary.csv",
        ),
        index=False,
    )

    # =========================================================================
    # Statistical significance testing across the 30 paired G-mean runs
    # =========================================================================
    (
        friedman_df,
        pairwise_wilcoxon_df,
        statistical_summary_df,
    ) = run_statistical_tests(
        result_df=result_df,
        output_dir=BASE_OUTPUT,
    )

    # =========================================================================
    # Statistical significance testing across the 30 paired runtime runs
    # =========================================================================
    (
        runtime_friedman_df,
        runtime_pairwise_df,
        runtime_statistical_summary_df,
    ) = run_runtime_tests(
        result_df=result_df,
        output_dir=BASE_OUTPUT,
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "RandomForest optimization ablation summary (30 paired runs)"
    )
    print(
        "=" * 78
    )
    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "Hyperparameter-tuning runtime summary"
    )
    print(
        "=" * 78
    )
    print(
        runtime_summary_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # G-mean boxplot - PDF
    # =========================================================================
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    result_df.boxplot(
        column="gmean",
        by="method",
        ax=ax,
    )

    ax.set_title(
        "G-mean Comparison across Optimization Methods"
    )
    fig.suptitle("")
    ax.set_ylabel(
        "G-mean"
    )
    ax.set_xlabel(
        "Method"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            BASE_OUTPUT,
            "mean_boxplot.pdf",
        ),
        format="pdf",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # Tuning-runtime boxplot - PDF
    # =========================================================================
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    result_df.boxplot(
        column="tuning_seconds",
        by="method",
        ax=ax,
    )

    ax.set_title(
        "Hyperparameter Tuning Runtime across Optimization Methods"
    )
    fig.suptitle("")
    ax.set_ylabel(
        "Runtime (seconds)"
    )
    ax.set_xlabel(
        "Method"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            BASE_OUTPUT,
            "tuning_runtime_boxplot.pdf",
        ),
        format="pdf",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"\nResults saved to: "
        f"{BASE_OUTPUT}"
    )
    print(
        f"Total runtime: "
        f"{elapsed:.2f} s "
        f"({elapsed / 60.0:.2f} min)"
    )

    return (
        result_df,
        summary_df,
        runtime_summary_df,
    )


if __name__ == "__main__":
    run_ablation()
