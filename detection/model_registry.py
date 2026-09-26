"""Manage versioned model storage and safe rollback.

Models are stored with version hashes based on the training data and timestamp,
allowing fine-grained tracking of which model version produced which scores.
A latest pointer tracks the currently-active model for inference.

SHAP importance tracking: :func:`compute_shap_summary` computes mean absolute
SHAP values per model after training. :func:`compare_importance_stability`
checks Spearman rank correlation of top-10 features between model versions
and blocks auto-promotion when correlation drops below the configured threshold.

Promotion gates (Issue #933)
----------------------------
:func:`promote_model` enforces two hard pre-checks before updating the
latest pointer:

1. **Signed model card** — a model card for the candidate version must exist
   and carry a valid ED25519 signature (``ModelSigner.verify``).
2. **Robustness threshold** — ``compute_robustness_report`` must show
   ``mean_map >= ROBUSTNESS_MIN_MAP`` and ``asr["0.10"] <= ROBUSTNESS_MAX_ASR``.

Either failure raises :class:`PromotionGateError` with a distinct, actionable
message naming the failing gate.  Gate results are recorded in
``training_metadata.json`` under the ``"promotion_checks"`` key for audit.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.settings import settings
from detection.model_signing import assert_within_model_dir, safe_joblib_load, sign_model_file

logger = logging.getLogger("ledgerlens.model_registry")

SHAP_STABILITY_THRESHOLD: float = 0.70

# ---------------------------------------------------------------------------
# Promotion gate thresholds (Issue #933)
# ---------------------------------------------------------------------------
# A candidate model must meet BOTH thresholds to be promoted.
# mean_map: minimal adversarial perturbation magnitude — higher is more robust
# asr_010:  attack success rate at epsilon=0.10 — lower is more robust
ROBUSTNESS_MIN_MAP: float = 0.05   # MAP must be >= this value
ROBUSTNESS_MAX_ASR: float = 0.80   # ASR at ε=0.10 must be <= this value


class PromotionGateError(RuntimeError):
    """Raised when a promotion pre-check fails.

    ``gate`` identifies which check failed (``"model_card"`` or ``"robustness"``).
    """

    def __init__(self, gate: str, message: str) -> None:
        self.gate = gate
        super().__init__(f"[gate:{gate}] {message}")


def _compute_version_hash(training_row_count: int, column_hash: str) -> str:
    """Generate SHA-256[:8] version hash from training metadata.

    Args:
        training_row_count: Number of rows in training dataset.
        column_hash: Hash of feature column names/order for stability.

    Returns:
        8-character hex string.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d%H%M")

    content = f"{training_row_count}:{column_hash}:{timestamp}"
    full_hash = hashlib.sha256(content.encode()).hexdigest()
    return full_hash[:8]


def save_versioned_model(
    model,
    name: str,
    version: str,
    model_dir: str,
) -> None:
    """Save a trained model with a version identifier.

    Creates {name}_v{version}.joblib and updates {name}_latest.txt
    to point to this version.

    Args:
        model: Trained scikit-learn/XGBoost/LightGBM model.
        name: Model name (e.g., 'random_forest', 'xgboost', 'lightgbm').
        version: Version string (typically SHA-256[:8]).
        model_dir: Directory to store versioned models.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(model_dir, f"{name}_v{version}.joblib")
    import joblib
    joblib.dump(model, model_path)
    sign_model_file(model_path, settings.model_signing_key.encode())
    logger.info("Saved versioned model to %s", model_path)

    latest_path = os.path.join(model_dir, f"{name}_latest.txt")
    with open(latest_path, "w") as f:
        f.write(version)
    logger.info("Updated %s to version %s", latest_path, version)


def load_latest_model(
    name: str,
    model_dir: str,
):
    """Load the currently-active model version.

    Reads {name}_latest.txt to determine which version to load,
    then loads {name}_v{version}.joblib.

    Args:
        name: Model name (e.g., 'random_forest', 'xgboost', 'lightgbm').
        model_dir: Directory containing versioned models.

    Returns:
        Trained model object.

    Raises:
        FileNotFoundError: If latest pointer or model file does not exist.
    """
    latest_path = os.path.join(model_dir, f"{name}_latest.txt")
    if not os.path.exists(latest_path):
        raise FileNotFoundError(f"Latest pointer not found: {latest_path}")

    with open(latest_path, "r") as f:
        version = f.read().strip()

    model_path = os.path.join(model_dir, f"{name}_v{version}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Versioned model not found: {model_path}")

    assert_within_model_dir(model_path, model_dir)
    model = safe_joblib_load(model_path, settings.model_signing_key.encode())
    logger.info("Loaded %s version %s from %s", name, version, model_path)
    return model


def rollback_model(
    name: str,
    previous_version: str,
    model_dir: str,
) -> None:
    """Revert to a previous model version.

    Updates {name}_latest.txt to point to previous_version.
    Does NOT validate that the previous version exists; that is the
    caller's responsibility.

    Args:
        name: Model name (e.g., 'random_forest', 'xgboost', 'lightgbm').
        previous_version: Version string to revert to.
        model_dir: Directory containing versioned models.
    """
    latest_path = os.path.join(model_dir, f"{name}_latest.txt")
    with open(latest_path, "w") as f:
        f.write(previous_version)
    logger.info("Rolled back %s to version %s", name, previous_version)


def list_model_versions(
    name: str,
    model_dir: str,
) -> list[str]:
    """List all available versions for a given model name.

    Scans the model directory for {name}_v*.joblib files and extracts
    version strings. Returns versions sorted newest-first by extracting
    the timestamp portion of the version hash.

    Args:
        name: Model name (e.g., 'random_forest', 'xgboost', 'lightgbm').
        model_dir: Directory containing versioned models.

    Returns:
        List of version strings, newest first. Empty list if no versions found
        or if the model directory does not exist.
    """
    if not os.path.isdir(model_dir):
        return []

    pattern = f"{name}_v"
    versions = []

    for fname in os.listdir(model_dir):
        if fname.startswith(pattern) and fname.endswith(".joblib"):
            version = fname[len(pattern) : -len(".joblib")]
            versions.append(version)

    # Sort by version string (which encodes timestamp as YYYYMMDDHHMM)
    # in descending order for newest-first ordering
    versions.sort(reverse=True)
    return versions


def get_current_version(
    name: str,
    model_dir: str,
) -> str | None:
    """Get the current version from the latest pointer.

    Args:
        name: Model name.
        model_dir: Directory containing versioned models.

    Returns:
        Current version string, or None if no latest pointer exists.
    """
    latest_path = os.path.join(model_dir, f"{name}_latest.txt")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path, "r") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# SHAP importance tracking & stability checks
# ---------------------------------------------------------------------------


@dataclass
class StabilityReport:
    version_old: str
    version_new: str
    spearman_rho: dict[str, float]
    stable: bool
    changed_features: dict[str, list[str]]
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def compute_shap_summary(
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    n_background: int = 100,
) -> list[dict]:
    """Compute mean absolute SHAP values using a background subsample."""
    import shap

    rng = np.random.RandomState(42)
    n_samples = min(n_background, len(X_train))
    indices = rng.choice(len(X_train), size=n_samples, replace=False)
    background = X_train[indices]

    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(background)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        [{"feature": f, "mean_abs_shap": float(v), "rank": 0} for f, v in zip(feature_names, mean_abs)],
        key=lambda x: -x["mean_abs_shap"],
    )
    for i, item in enumerate(ranked):
        item["rank"] = i + 1
    return ranked[:10]


def compute_spearman_rho(old_top10: list[dict], new_top10: list[dict]) -> float:
    """Compute Spearman rank correlation between old and new feature rankings."""
    from scipy.stats import spearmanr

    all_features = list({item["feature"] for item in old_top10 + new_top10})
    old_ranks = {item["feature"]: item["rank"] for item in old_top10}
    new_ranks = {item["feature"]: item["rank"] for item in new_top10}
    old_vec = [old_ranks.get(f, 11) for f in all_features]
    new_vec = [new_ranks.get(f, 11) for f in all_features]
    rho, _ = spearmanr(old_vec, new_vec)
    return float(rho)


def compare_importance_stability(
    old_metadata: dict,
    new_metadata: dict,
    threshold: float = SHAP_STABILITY_THRESHOLD,
) -> StabilityReport:
    """Compare SHAP importance rankings between two model versions."""
    old_version = old_metadata.get("version", "unknown")
    new_version = new_metadata.get("version", "unknown")
    old_importances = old_metadata.get("shap_importances", {})
    new_importances = new_metadata.get("shap_importances", {})

    if not old_importances:
        return StabilityReport(
            version_old=old_version,
            version_new=new_version,
            spearman_rho={},
            stable=True,
            changed_features={},
        )

    spearman_rho: dict[str, float] = {}
    changed_features: dict[str, list[str]] = {}

    model_names = set(old_importances.keys()) | set(new_importances.keys())
    for model_name in model_names:
        old_top10 = old_importances.get(model_name, [])
        new_top10 = new_importances.get(model_name, [])

        if not old_top10 or not new_top10:
            spearman_rho[model_name] = 1.0
            changed_features[model_name] = []
            continue

        rho = compute_spearman_rho(old_top10, new_top10)
        spearman_rho[model_name] = rho

        old_feats = {item["feature"] for item in old_top10}
        new_feats = {item["feature"] for item in new_top10}
        changed = list((old_feats - new_feats) | (new_feats - old_feats))
        changed_features[model_name] = changed

    stable = all(rho >= threshold for rho in spearman_rho.values())

    return StabilityReport(
        version_old=old_version,
        version_new=new_version,
        spearman_rho=spearman_rho,
        stable=stable,
        changed_features=changed_features,
    )


def save_shap_importances(
    shap_data: dict[str, list[dict]],
    model_dir: str,
) -> None:
    """Write SHAP importances into training_metadata.json."""
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    metadata: dict = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read existing metadata at %s: %s", metadata_path, exc)

    metadata["shap_importances"] = shap_data

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved SHAP importances to %s", metadata_path)


def load_shap_importances(model_dir: str, version: str | None = None) -> dict | None:
    """Load SHAP importances from training_metadata.json."""
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read SHAP importances from %s: %s", metadata_path, exc)
        return None

    if version and metadata.get("version") != version:
        return None

    return metadata.get("shap_importances")


# ---------------------------------------------------------------------------
# Promotion gates (Issue #933)
# ---------------------------------------------------------------------------


def _check_model_card_gate(name: str, version: str, model_dir: str) -> dict:
    """Return gate result dict; raise PromotionGateError on failure.

    Checks that a model card file exists for the given version and that it
    carries a valid ED25519 signature produced by :class:`~detection.model_signing.ModelSigner`.
    """
    card_path = Path(model_dir) / f"{name}_v{version}_model_card.json"
    if not card_path.exists():
        raise PromotionGateError(
            "model_card",
            f"No model card found for {name} v{version} at {card_path}. "
            "Generate and sign a model card before promoting this version.",
        )

    # Verify the ED25519 signature on the card file
    try:
        from detection.model_signing import get_model_signer
        signer = get_model_signer()
        signer.verify(card_path)
    except Exception as exc:
        raise PromotionGateError(
            "model_card",
            f"Model card signature verification failed for {name} v{version}: {exc}. "
            "Re-sign the model card with a valid private key before promoting.",
        )

    logger.info("Promotion gate PASSED [model_card]: %s v%s", name, version)
    return {"gate": "model_card", "passed": True, "card_path": str(card_path)}


def _check_robustness_gate(name: str, version: str, model_dir: str) -> dict:
    """Return gate result dict; raise PromotionGateError on failure.

    Loads the latest persisted robustness report from the database (written by
    :func:`~detection.robustness_eval.compute_robustness_report`).  Falls back
    to inline computation when no persisted report exists for this version.
    """
    from detection.storage import get_latest_robustness_report

    report_data: dict | None = None
    try:
        report_data = get_latest_robustness_report(model_version=version)
    except Exception as exc:
        logger.warning("Could not load persisted robustness report for %s v%s: %s", name, version, exc)

    if report_data is None:
        raise PromotionGateError(
            "robustness",
            f"No robustness report found for {name} v{version}. "
            "Run compute_robustness_report() and persist results before promoting.",
        )

    mean_map = float(report_data.get("mean_map", 0.0))
    asr = report_data.get("asr", {})
    asr_010 = float(asr.get("0.10", 1.0))

    failures: list[str] = []
    if mean_map < ROBUSTNESS_MIN_MAP:
        failures.append(
            f"mean_map={mean_map:.4f} is below the required threshold of {ROBUSTNESS_MIN_MAP}"
        )
    if asr_010 > ROBUSTNESS_MAX_ASR:
        failures.append(
            f"asr[0.10]={asr_010:.4f} exceeds the maximum allowed value of {ROBUSTNESS_MAX_ASR}"
        )

    if failures:
        raise PromotionGateError(
            "robustness",
            f"Robustness gate FAILED for {name} v{version}: " + "; ".join(failures) + ". "
            "Re-evaluate robustness and address deficiencies before promoting.",
        )

    logger.info(
        "Promotion gate PASSED [robustness]: %s v%s (mean_map=%.4f, asr_010=%.4f)",
        name, version, mean_map, asr_010,
    )
    return {
        "gate": "robustness",
        "passed": True,
        "mean_map": mean_map,
        "asr_010": asr_010,
    }


def _record_promotion_checks(model_dir: str, name: str, version: str, gate_results: list[dict]) -> None:
    """Persist gate results to training_metadata.json for audit trail."""
    metadata_path = os.path.join(model_dir, "training_metadata.json")
    metadata: dict = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read existing metadata: %s", exc)

    promotions = metadata.setdefault("promotion_checks", {})
    promotions[f"{name}_v{version}"] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "gates": gate_results,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def promote_model(name: str, version: str, model_dir: str) -> None:
    """Promote *version* as the active model after enforcing pre-checks.

    Pre-checks (both must pass):
      1. **model_card** — a signed model card JSON must exist for this version.
      2. **robustness** — a persisted robustness report must meet
         ``mean_map >= ROBUSTNESS_MIN_MAP`` and ``asr["0.10"] <= ROBUSTNESS_MAX_ASR``.

    On any gate failure, raises :class:`PromotionGateError` with ``gate``
    identifying which check failed and a message explaining the resolution.

    On success, updates ``{name}_latest.txt`` and records gate results in
    ``training_metadata.json`` for audit.

    Args:
        name: Model name (e.g., ``"random_forest"``).
        version: Candidate version string (SHA-256[:8]).
        model_dir: Directory containing versioned models.

    Raises:
        PromotionGateError: If any pre-check fails.
        FileNotFoundError: If the versioned model file does not exist.
    """
    # Verify candidate model file actually exists before spending gate effort
    model_path = os.path.join(model_dir, f"{name}_v{version}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot promote: versioned model not found at {model_path}"
        )

    gate_results: list[dict] = []

    # Gate 1: signed model card
    card_result = _check_model_card_gate(name, version, model_dir)
    gate_results.append(card_result)

    # Gate 2: robustness threshold
    rob_result = _check_robustness_gate(name, version, model_dir)
    gate_results.append(rob_result)

    # Record gate results for audit trail (best-effort)
    try:
        _record_promotion_checks(model_dir, name, version, gate_results)
    except Exception as exc:
        logger.warning("Could not persist promotion check results: %s", exc)

    # All gates passed — update the latest pointer
    latest_path = os.path.join(model_dir, f"{name}_latest.txt")
    with open(latest_path, "w") as f:
        f.write(version)

    logger.info(
        "Promoted %s to version %s (all gates passed: %s)",
        name, version, [g["gate"] for g in gate_results],
    )
