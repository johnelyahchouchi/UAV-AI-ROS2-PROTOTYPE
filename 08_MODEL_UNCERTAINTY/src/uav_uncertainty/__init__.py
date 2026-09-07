"""Detector-independent uncertainty analysis for the UAV perception prototype."""

from .analysis import ImageAnalysis, analyze_image
from .detection import Detection
from .metrics import TargetMetrics
from .methods import MCDropoutCapability, mc_dropout_capability
from .mc_dropout_v2 import (
    MCDOArchitectureReport,
    MCDOFrameAnalysis,
    MCDOPassRecord,
    MCDOTargetCluster,
    MCDOTargetResult,
    MCDOV2Config,
    MCDOV2Error,
    activate_mcdo_inference,
    calculate_mcdo_target,
    cluster_mcdo_samples,
    run_mcdo_frame,
    validate_mcdo_architecture,
    validate_mcdo_inference_state,
)
from .perturbations import PerturbationConfig

__all__ = [
    "Detection",
    "ImageAnalysis",
    "MCDropoutCapability",
    "MCDOArchitectureReport",
    "MCDOFrameAnalysis",
    "MCDOPassRecord",
    "MCDOTargetCluster",
    "MCDOTargetResult",
    "MCDOV2Config",
    "MCDOV2Error",
    "PerturbationConfig",
    "TargetMetrics",
    "analyze_image",
    "activate_mcdo_inference",
    "calculate_mcdo_target",
    "cluster_mcdo_samples",
    "mc_dropout_capability",
    "run_mcdo_frame",
    "validate_mcdo_architecture",
    "validate_mcdo_inference_state",
]
