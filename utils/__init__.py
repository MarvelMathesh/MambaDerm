"""MambaDerm Utils Package."""

from .metrics import (
    compute_pauc, 
    pAUCMetric,
    compute_balanced_accuracy,
    compute_macro_f1,
    compute_per_class_accuracy,
    MultiClassMetrics,
)
from .losses import (
    FocalLoss, 
    AsymmetricLoss, 
    LabelSmoothingBCE,
    AUCMaxLoss,
    MultiObjectiveLoss,
    MultiClassFocalLoss,
    LabelSmoothingCrossEntropy,
    MultiClassMultiObjectiveLoss,
)
from .scheduler import get_cosine_schedule_with_warmup
from .tta import TTAWrapper, TTAInference, get_tta_predictions
from .uncertainty import (
    MCDropout,
    TemperatureScaling,
    OODDetector,
    SafePredictor,
    compute_ece,
)
from .export import (
    export_torchscript,
    export_onnx,
    quantize_dynamic,
    benchmark_inference,
    InferencePipeline,
)

__all__ = [
    # Metrics (binary)
    "compute_pauc",
    "pAUCMetric",
    # Metrics (multi-class)
    "compute_balanced_accuracy",
    "compute_macro_f1", 
    "compute_per_class_accuracy",
    "MultiClassMetrics",
    # Losses (binary)
    "FocalLoss",
    "AsymmetricLoss",
    "LabelSmoothingBCE",
    "AUCMaxLoss",
    "MultiObjectiveLoss",
    # Losses (multi-class)
    "MultiClassFocalLoss",
    "LabelSmoothingCrossEntropy",
    "MultiClassMultiObjectiveLoss",
    # Scheduler
    "get_cosine_schedule_with_warmup",
    # TTA
    "TTAWrapper",
    "TTAInference",
    "get_tta_predictions",
    # Uncertainty
    "MCDropout",
    "TemperatureScaling",
    "OODDetector",
    "SafePredictor",
    "compute_ece",
    # Export
    "export_torchscript",
    "export_onnx",
    "quantize_dynamic",
    "benchmark_inference",
    "InferencePipeline",
]
