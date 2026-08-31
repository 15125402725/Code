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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# Publication-quality figure style
# =============================================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from scipy.stats import friedmanchisquare, wilcoxon

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, make_scorer
from imblearn.metrics import geometric_mean_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# =============================================================================
# Global configuration
# =============================================================================
BASE_OUTPUT = "output_ablation_rf_journal_clean_v11"
os.makedirs(BASE_OUTPUT, exist_ok=True)


# =============================================================================
# Figure saving utility
# Save only PDF and TIFF formats
# =============================================================================
def save_figure_pdf_tiff(fig, filepath):
    """
    PDF: vector format
    TIFF: 600 dpi publication format
    """
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

BASE_RANDOM_STATE = 42
N_REPEATS = 30

CV_FOLDS = 5
LASSO_INNER_CV_FOLDS = 5
LASSO_FIXED_ALPHA = None

N_JOBS = max(1, min(8, os.cpu_count() or 1))

PSO_PARTICLES = 10
PSO_ITERATIONS = 15

# CPSO evaluates all particles once before the iterative updates,
# then evaluates all particles once per iteration.
PSO_CANDIDATE_EVALUATIONS = PSO_PARTICLES * (PSO_ITERATIONS + 1)

CONFIDENCE = 0.95

# Primary optimization objective for imbalanced classification
gmean_scorer = make_scorer(
    geometric_mean_score
)



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
        initial_positions=None,
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
        self.initial_positions = initial_positions

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

        # ---------------------------------------------------------------------
        # Elite injection
        # For Grid-CPSO, the best Grid solution is inserted directly into
        # the initial swarm. CPSO-only leaves initial_positions=None and
        # therefore retains pure chaotic initialization.
        # ---------------------------------------------------------------------
        if self.initial_positions is not None:
            elite_positions = np.atleast_2d(
                np.asarray(
                    self.initial_positions,
                    dtype=float,
                )
            )

            if elite_positions.shape[1] != self.dim:
                raise ValueError(
                    "Elite initialization dimension does not match "
                    "the CPSO search-space dimension."
                )

            n_elites = min(
                len(elite_positions),
                self.n_particles,
            )

            positions[:n_elites] = np.clip(
                elite_positions[:n_elites],
                self.lb,
                self.ub,
            )
        else:
            n_elites = 0

        velocities = self.rng.uniform(
            low=-(self.ub - self.lb) / 2.0,
            high=(self.ub - self.lb) / 2.0,
            size=(self.n_particles, self.dim),
        )

        # Start injected elite solutions without an arbitrary initial velocity.
        if n_elites > 0:
            velocities[:n_elites] = 0.0

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
                f"best CV G-mean="
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


def _neighbor_midpoint_bounds(
    values,
    center_value,
):
    """
    Construct a local interval around the Grid optimum from adjacent Grid points.

    Example:
        values = [100, 400, 800], center = 400
        -> lower = (100 + 400) / 2 = 250
        -> upper = (400 + 800) / 2 = 600

    Boundary Grid points use the corresponding global search boundary.
    """
    numeric_values = sorted(
        float(v)
        for v in values
    )
    center = float(center_value)

    distances = np.abs(
        np.asarray(numeric_values)
        - center
    )
    center_idx = int(
        np.argmin(distances)
    )

    if not np.isclose(
        numeric_values[center_idx],
        center,
    ):
        raise ValueError(
            f"Grid center value {center_value} was not found "
            f"in candidate values {values}."
        )

    if center_idx == 0:
        low = numeric_values[0]
    else:
        low = (
            numeric_values[center_idx - 1]
            + center
        ) / 2.0

    if center_idx == len(numeric_values) - 1:
        high = numeric_values[-1]
    else:
        high = (
            center
            + numeric_values[center_idx + 1]
        ) / 2.0

    return float(low), float(high)


def build_pso_specs(
    param_grid,
    center_params=None,
    local=False,
):
    """
    Build CPSO search specifications.

    CPSO-only:
        local=False -> search the full parameter range.

    Grid-CPSO:
        local=True -> numeric parameters are refined inside the interval defined
        by neighboring Grid points around the Grid optimum; categorical
        parameters are refined within the Grid optimum and adjacent categories.
    """
    specs = []
    bounds = []

    for key, values in param_grid.items():
        values = list(values)

        if _is_pure_numeric(values):
            if (
                local
                and center_params is not None
            ):
                low, high = (
                    _neighbor_midpoint_bounds(
                        values,
                        center_params[key],
                    )
                )
            else:
                low = float(min(values))
                high = float(max(values))

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
            # Categorical/discrete parameters use encoded category indices.
            if (
                local
                and center_params is not None
            ):
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


def encode_params_to_vector(
    params,
    specs,
):
    """
    Encode a concrete Grid parameter configuration into the CPSO vector space.

    This is used to inject the best Grid solution as an elite particle into
    the initial Grid-CPSO swarm.
    """
    vec = []

    for spec in specs:
        key = spec["key"]
        value = params[key]

        if spec["kind"] == "numeric":
            encoded = float(value)
        else:
            encoded = float(
                spec["values"].index(
                    value
                )
            )

        vec.append(encoded)

    return np.asarray(
        vec,
        dtype=float,
    )


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
        scoring=gmean_scorer,
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
        f"    Grid best CV G-mean="
        f"{grid.best_score_:.4f}; "
        f"tuning time="
        f"{elapsed:.2f} s"
    )

    return (
        grid.best_params_.copy(),
        float(grid.best_score_),
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

    # CPSO-only performs an independent search over the full parameter range.
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
            scoring=gmean_scorer,
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
        initial_positions=None,
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
        f"    CPSO-only best CV G-mean="
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
    grid_params=None,
    grid_score=None,
    grid_time=None,
):
    """
    Coarse-to-fine Grid-CPSO hybrid optimization.

    Efficiency improvement:
        The Grid-only result from the SAME repeat/seed can be passed into this
        function. The identical GridSearchCV is then reused rather than executed
        a second time. This changes runtime only; it does not change the search
        space, CV splits, scoring rule, or selected Grid solution.

    Stage 2:
        CPSO constructs a local search region from neighboring Grid points.
        The best Grid configuration is injected as an elite initial particle,
        while the remaining particles retain chaotic initialization.
    """
    total_start_time = time.perf_counter()

    if grid_params is None or grid_score is None:
        grid_params, grid_score, actual_grid_time = get_grid_params(
            base_pipeline,
            X,
            y,
            param_grid,
            random_state,
        )
        grid_time = actual_grid_time
        grid_reused = False
    else:
        grid_params = grid_params.copy()
        grid_score = float(grid_score)
        grid_time = 0.0 if grid_time is None else float(grid_time)
        grid_reused = True
        print(
            f"    Reusing Grid-only result: best CV G-mean={grid_score:.4f}; "
            f"GridSearchCV is not repeated."
        )

    cpso_start_time = time.perf_counter()

    specs, bounds = build_pso_specs(
        param_grid,
        center_params=grid_params,
        local=True,
    )

    elite_vec = encode_params_to_vector(
        grid_params,
        specs,
    )

    def objective(vec):
        candidate_params = decode_pso_vector(
            vec,
            specs,
        )

        candidate = clone(base_pipeline)
        candidate.set_params(**candidate_params)

        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=make_cv(random_state),
            scoring=gmean_scorer,
            n_jobs=1,
            error_score="raise",
        )
        return float(np.mean(scores))

    pso = SimplePSO(
        objective_func=objective,
        bounds=bounds,
        n_particles=PSO_PARTICLES,
        max_iter=PSO_ITERATIONS,
        use_chaotic_w=True,
        n_jobs=N_JOBS,
        random_state=random_state,
        initial_positions=[elite_vec],
    )

    best_vec, cpso_score = pso.optimize()

    cpso_time = time.perf_counter() - cpso_start_time
    cpso_params = decode_pso_vector(best_vec, specs)

    if cpso_score > grid_score:
        final_params = cpso_params
        final_score = float(cpso_score)
        source = "CPSO_refinement"
    else:
        final_params = grid_params
        final_score = float(grid_score)
        source = "Grid_retained"

    # This is the actual extra wall-clock time spent by the hybrid call.
    # If Grid was reused, it is essentially the CPSO refinement time.
    incremental_wall_time = time.perf_counter() - total_start_time

    # For methodological runtime reporting, Grid-CPSO still conceptually costs
    # Grid + CPSO. We therefore retain the original Grid time in this quantity.
    method_equivalent_total_time = float(grid_time) + float(cpso_time)

    print(
        f"    Grid+CPSO selected CV G-mean={final_score:.4f} ({source})"
    )
    print(
        f"    Grid reused={grid_reused}; "
        f"Grid stage time={grid_time:.2f} s; "
        f"CPSO refinement time={cpso_time:.2f} s; "
        f"method-equivalent total={method_equivalent_total_time:.2f} s; "
        f"actual incremental wall time={incremental_wall_time:.2f} s"
    )

    return (
        final_params,
        final_score,
        method_equivalent_total_time,
        float(grid_time),
        float(cpso_time),
        source,
        bool(grid_reused),
        float(incremental_wall_time),
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



def run_statistical_tests(
    result_df,
    output_dir,
    value_col,
    metric_label,
    file_prefix,
):
    """
    Paired nonparametric comparison across repeated runs.

    Statistical workflow:
        1. Friedman test for the overall three-method comparison.
        2. If the Friedman test is significant, paired Wilcoxon signed-rank
           post-hoc tests comparing Grid-CPSO with each component method.
        3. Holm correction for the two post-hoc comparisons.

    This function is used separately for:
        - tuning_cv_gmean: optimization-level endpoint (Best CV G-mean)
        - gmean: independent test-set endpoint (Test G-mean)
    """
    required_methods = [
        "Grid_Search",
        "CPSO_only",
        "Grid_CPSO",
    ]

    paired = result_df.pivot_table(
        index=["repeat", "seed"],
        columns="method",
        values=value_col,
        aggfunc="first",
    )

    missing_methods = [
        method
        for method in required_methods
        if method not in paired.columns
    ]
    if missing_methods:
        raise ValueError(
            f"{metric_label} statistical testing cannot proceed because "
            f"methods are missing: {missing_methods}"
        )

    paired = paired[required_methods].dropna().copy()

    if len(paired) < 2:
        raise ValueError(
            f"Insufficient paired runs for {metric_label} statistical testing."
        )

    paired.reset_index().to_csv(
        os.path.join(
            output_dir,
            f"{file_prefix}_paired_runs_for_statistics.csv",
        ),
        index=False,
    )

    grid = paired["Grid_Search"].to_numpy(dtype=float)
    cpso = paired["CPSO_only"].to_numpy(dtype=float)
    hybrid = paired["Grid_CPSO"].to_numpy(dtype=float)

    # Overall Friedman test.
    friedman_stat, friedman_p = friedmanchisquare(
        grid,
        cpso,
        hybrid,
    )

    friedman_df = pd.DataFrame(
        [
            {
                "metric": metric_label,
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
            f"friedman_test_{file_prefix}.csv",
        ),
        index=False,
    )

    # Post-hoc comparisons are performed only after a significant Friedman test.
    comparisons = [
        ("Grid_CPSO", "Grid_Search", hybrid, grid),
        ("Grid_CPSO", "CPSO_only", hybrid, cpso),
    ]

    pairwise_rows = []

    if friedman_p < 0.05:
        for method_a, method_b, values_a, values_b in comparisons:
            differences = values_a - values_b

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
                    "metric": metric_label,
                    "test": "Wilcoxon signed-rank",
                    "method_A": method_a,
                    "method_B": method_b,
                    "n_paired_runs": len(paired),
                    "mean_method_A": float(np.mean(values_a)),
                    "mean_method_B": float(np.mean(values_b)),
                    "mean_difference_A_minus_B": float(np.mean(differences)),
                    "n_A_greater_B": int(np.sum(differences > 0)),
                    "n_A_equal_B": int(np.sum(np.isclose(differences, 0.0))),
                    "n_A_less_B": int(np.sum(differences < 0)),
                    "statistic": float(statistic),
                    "p_value_raw": float(p_value),
                }
            )

        pairwise_df = pd.DataFrame(pairwise_rows)
        pairwise_df["p_value_holm"] = holm_adjust_pvalues(
            pairwise_df["p_value_raw"].to_numpy()
        )
        pairwise_df["significant_holm_0.05"] = (
            pairwise_df["p_value_holm"] < 0.05
        )
    else:
        pairwise_df = pd.DataFrame(
            columns=[
                "metric",
                "test",
                "method_A",
                "method_B",
                "n_paired_runs",
                "mean_method_A",
                "mean_method_B",
                "mean_difference_A_minus_B",
                "n_A_greater_B",
                "n_A_equal_B",
                "n_A_less_B",
                "statistic",
                "p_value_raw",
                "p_value_holm",
                "significant_holm_0.05",
            ]
        )

    pairwise_df.to_csv(
        os.path.join(
            output_dir,
            f"wilcoxon_pairwise_{file_prefix}_holm.csv",
        ),
        index=False,
    )

    print("\n" + "=" * 78)
    print(f"Statistical testing: {metric_label}")
    print("=" * 78)
    print(
        f"Friedman: statistic={friedman_stat:.6f}, "
        f"p={friedman_p:.6g}"
    )

    if friedman_p < 0.05:
        print("\nPost-hoc paired Wilcoxon tests with Holm correction:")
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
            ].to_string(index=False)
        )
    else:
        print(
            "\nFriedman test was not significant; "
            "post-hoc Wilcoxon comparisons were not interpreted."
        )

    return friedman_df, pairwise_df


# =============================================================================
# Main ablation
# =============================================================================
def run_ablation():
    start_time = (
        time.perf_counter()
    )

    # -------------------------------------------------------------------------
    # Coarse Grid used BOTH by Grid-only and by Stage 1 of Grid-CPSO.
    # 3 x 3 x 3 x 2 x 2 = 108 candidate configurations.
    #
    # Grid-only and the Grid stage of Grid-CPSO use exactly the same grid.
    # The hybrid then uses CPSO for neighboring-grid local refinement.
    # -------------------------------------------------------------------------
    param_grid = {
        "classifier__n_estimators": [
            100,
            400,
            800,
        ],
        "classifier__max_depth": [
            5,
            15,
            30,
        ],
        "classifier__min_samples_split": [
            2,
            10,
            20,
        ],
        "classifier__min_samples_leaf": [
            1,
            10,
        ],
        "classifier__max_features": [
            "sqrt",
            None,
        ],
    }

    method_names = [
        "Grid_Search",
        "CPSO_only",
        "Grid_CPSO",
    ]

    results = []

    # Track whether hybrid optimization improves over Grid.
    hybrid_source_counter = {
        "CPSO_refinement": 0,
        "Grid_retained": 0,
    }

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

        # Grid-only and Grid-CPSO use the same Grid stage for this repeat.
        # Cache it once and reuse it in the hybrid arm.
        cached_grid_params = None
        cached_grid_score = None
        cached_grid_time = None

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
            grid_result_reused = False
            hybrid_incremental_seconds = np.nan

            if (
                method_name
                == "Grid_Search"
            ):
                (
                    params,
                    tuning_cv_gmean,
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
                optimization_source = (
                    "Grid_only"
                )

                cached_grid_params = params.copy()
                cached_grid_score = float(tuning_cv_gmean)
                cached_grid_time = float(tuning_seconds)


            elif (
                method_name
                == "CPSO_only"
            ):
                (
                    params,
                    tuning_cv_gmean,
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
                optimization_source = (
                    "CPSO_only"
                )

            else:
                (
                    params,
                    tuning_cv_gmean,
                    tuning_seconds,
                    grid_stage_seconds,
                    cpso_stage_seconds,
                    optimization_source,
                    grid_result_reused,
                    hybrid_incremental_seconds,
                ) = get_grid_pso_params(
                    base_pipeline,
                    X_train,
                    y_train,
                    param_grid,
                    seed,
                    grid_params=cached_grid_params,
                    grid_score=cached_grid_score,
                    grid_time=cached_grid_time,
                )

                hybrid_source_counter[
                    optimization_source
                ] += 1

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
                "tuning_cv_gmean": (
                    tuning_cv_gmean
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
                "optimization_source": (
                    optimization_source
                ),
                "grid_result_reused": (
                    grid_result_reused
                ),
                "hybrid_incremental_seconds": (
                    hybrid_incremental_seconds
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
    # Hybrid refinement-source statistics
    # =========================================================================
    hybrid_source_df = pd.DataFrame(
        [
            {
                "optimization_source": source,
                "count": count,
                "proportion": (
                    count / N_REPEATS
                ),
            }
            for source, count
            in hybrid_source_counter.items()
        ]
    )

    hybrid_source_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "hybrid_refinement_source_summary.csv",
        ),
        index=False,
    )

    # =========================================================================
    # Optimization-budget summary
    # Candidate evaluations are reported separately from CV model fits.
    # =========================================================================
    n_grid_candidates = int(
        np.prod(
            [
                len(values)
                for values
                in param_grid.values()
            ]
        )
    )

    optimization_budget_df = pd.DataFrame(
        [
            {
                "method": "Grid_Search",
                "grid_candidate_evaluations": n_grid_candidates,
                "cpso_candidate_evaluations": 0,
                "total_candidate_evaluations": n_grid_candidates,
                "cv_folds": CV_FOLDS,
                "estimated_cv_model_fits": (
                    n_grid_candidates
                    * CV_FOLDS
                ),
            },
            {
                "method": "CPSO_only",
                "grid_candidate_evaluations": 0,
                "cpso_candidate_evaluations": PSO_CANDIDATE_EVALUATIONS,
                "total_candidate_evaluations": PSO_CANDIDATE_EVALUATIONS,
                "cv_folds": CV_FOLDS,
                "estimated_cv_model_fits": (
                    PSO_CANDIDATE_EVALUATIONS
                    * CV_FOLDS
                ),
            },
            {
                "method": "Grid_CPSO",
                "grid_candidate_evaluations": n_grid_candidates,
                "cpso_candidate_evaluations": PSO_CANDIDATE_EVALUATIONS,
                "total_candidate_evaluations": (
                    n_grid_candidates
                    + PSO_CANDIDATE_EVALUATIONS
                ),
                "cv_folds": CV_FOLDS,
                "estimated_cv_model_fits": (
                    (
                        n_grid_candidates
                        + PSO_CANDIDATE_EVALUATIONS
                    )
                    * CV_FOLDS
                ),
            },
        ]
    )

    optimization_budget_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "optimization_budget_summary.csv",
        ),
        index=False,
    )

    experiment_configuration_df = pd.DataFrame(
        [
            {"setting": "repeats", "value": N_REPEATS},
            {"setting": "cv_folds", "value": CV_FOLDS},
            {"setting": "grid_candidates", "value": n_grid_candidates},
            {"setting": "cpso_particles", "value": PSO_PARTICLES},
            {"setting": "cpso_iterations", "value": PSO_ITERATIONS},
            {
                "setting": "cpso_candidate_evaluations",
                "value": PSO_CANDIDATE_EVALUATIONS,
            },
        ]
    )
    experiment_configuration_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "experiment_configuration.csv",
        ),
        index=False,
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
    # Concise scientific summary
    # Keep optimization-level and independent test-set endpoints separate.
    # Runtime is intentionally excluded from journal statistical outputs.
    # =========================================================================
    summary_rows = []

    cp_cols = [
        col
        for col in result_df.columns
        if col.startswith("cp_")
    ]

    for method in method_names:
        sub = result_df[
            result_df["method"] == method
        ]

        row = {
            "method": method,
            "best_cv_gmean_mean": sub["tuning_cv_gmean"].mean(),
            "best_cv_gmean_sd": sub["tuning_cv_gmean"].std(ddof=1),
            "best_cv_gmean_min": sub["tuning_cv_gmean"].min(),
            "best_cv_gmean_max": sub["tuning_cv_gmean"].max(),
            "test_gmean_mean": sub["gmean"].mean(),
            "test_gmean_sd": sub["gmean"].std(ddof=1),
            "test_gmean_min": sub["gmean"].min(),
            "test_gmean_max": sub["gmean"].max(),
        }

        for col in cp_cols:
            row[
                col.replace("cp_", "cp_mean_")
            ] = sub[col].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).round(4)

    summary_df.to_csv(
        os.path.join(
            BASE_OUTPUT,
            "ablation_summary.csv",
        ),
        index=False,
    )

    # =========================================================================
    # Statistical analysis 1: optimization-level endpoint
    # Best CV G-mean is the objective directly optimized by all three methods.
    # =========================================================================
    (
        cv_friedman_df,
        cv_pairwise_df,
    ) = run_statistical_tests(
        result_df=result_df,
        output_dir=BASE_OUTPUT,
        value_col="tuning_cv_gmean",
        metric_label="Best CV G-mean",
        file_prefix="cv_gmean",
    )

    # =========================================================================
    # Statistical analysis 2: independent test-set endpoint
    # This evaluates whether optimization gains translate to generalization.
    # =========================================================================
    (
        test_friedman_df,
        test_pairwise_df,
    ) = run_statistical_tests(
        result_df=result_df,
        output_dir=BASE_OUTPUT,
        value_col="gmean",
        metric_label="Test G-mean",
        file_prefix="test_gmean",
    )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "RandomForest Grid-CPSO ablation: 108-Grid + 10-particle/15-iteration CPSO (30 paired runs)"
    )
    print(
        "=" * 78
    )
    print(
        summary_df.to_string(
            index=False
        )
    )


    # =========================================================================
    # Best CV G-mean boxplot: optimization-level comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    result_df.boxplot(
        column="tuning_cv_gmean",
        by="method",
        ax=ax,
    )

    ax.set_title(
        "Best CV G-mean across Optimization Methods",
        fontsize=15,
        fontweight="bold",
    )
    fig.suptitle("")
    ax.set_ylabel("Best CV G-mean", fontsize=15)
    ax.set_xlabel("Method", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    fig.tight_layout()

    save_figure_pdf_tiff(
        fig,
        os.path.join(
            BASE_OUTPUT,
            "cv_gmean_boxplot",
        ),
    )
    plt.close(fig)

    # =========================================================================
    # Test G-mean boxplot: independent generalization comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    result_df.boxplot(
        column="gmean",
        by="method",
        ax=ax,
    )

    ax.set_title(
        "Test G-mean across Optimization Methods",
        fontsize=15,
        fontweight="bold",
    )
    fig.suptitle("")
    ax.set_ylabel("Test G-mean", fontsize=15)
    ax.set_xlabel("Method", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    fig.tight_layout()

    save_figure_pdf_tiff(
        fig,
        os.path.join(
            BASE_OUTPUT,
            "test_gmean_boxplot",
        ),
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
    )


if __name__ == "__main__":
    run_ablation()
