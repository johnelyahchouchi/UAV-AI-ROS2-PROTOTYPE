"""Detector-independent MC Dropout V2 analysis and model-state validation.

V2 repeats one unchanged frame while a validated detector's six late-head
``Dropout2d`` layers are active.  It is separate from V1 input perturbation and
does not produce a calibrated correctness probability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
from statistics import fmean, pstdev
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .detection import BoundingBox, Detection
from .matching import intersection_over_union


METHOD_ID = "v2_mc_dropout_model_uncertainty"
VALIDATED_DROPOUT_COUNT = 6
VALIDATED_DROPOUT_PROBABILITY = 0.20
VALIDATED_DETECTION_SCALES = 3

Image = NDArray[np.uint8]


class MCDOV2Error(RuntimeError):
    """Raised when V2 cannot run without violating its validated method."""


@dataclass(frozen=True)
class MCDOV2Config:
    """Fixed architecture expectations and configurable inference thresholds."""

    sample_count: int = 20
    match_iou: float = 0.50
    confidence: float = 0.25
    nms_iou: float = 0.45
    image_size: int = 640
    dropout_count: int = VALIDATED_DROPOUT_COUNT
    dropout_probability: float = VALIDATED_DROPOUT_PROBABILITY

    def __post_init__(self) -> None:
        numeric = (
            self.match_iou,
            self.confidence,
            self.nms_iou,
            self.dropout_probability,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("V2 numeric configuration must be finite")
        if not 1 <= self.sample_count <= 100:
            raise ValueError("V2 sample_count must be between 1 and 100")
        if not 0.0 < self.match_iou <= 1.0:
            raise ValueError("V2 match_iou must be greater than 0 and at most 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("V2 confidence must be between 0 and 1")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ValueError("V2 nms_iou must be between 0 and 1")
        if self.image_size <= 0 or self.image_size > 8192:
            raise ValueError("V2 image_size must be between 1 and 8192")
        if self.dropout_count != VALIDATED_DROPOUT_COUNT:
            raise ValueError("Validated V2 requires exactly 6 Dropout2d layers")
        if not math.isclose(
            self.dropout_probability,
            VALIDATED_DROPOUT_PROBABILITY,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Validated V2 requires Dropout2d p=0.20")


@dataclass(frozen=True)
class MCDOArchitectureReport:
    """Verified V2 detection-head architecture and inference state."""

    dropout_count: int
    dropout_probability: float
    batchnorm_count: int
    placements: tuple[str, ...]
    all_dropout_active: bool
    all_batchnorm_eval: bool
    model_eval: bool


@dataclass(frozen=True)
class AxisVariation:
    """Population standard deviations for two spatial axes."""

    x: float
    y: float


@dataclass(frozen=True)
class MCDOPassRecord:
    """Per-object hypotheses retained for one stochastic forward pass."""

    pass_index: int
    winner_class_id: int
    winner_class_name: str
    winner_confidence: float
    class_hypotheses: tuple[Detection, ...]
    evidence_share: Mapping[str, float] | None
    representative_bbox_xyxy: BoundingBox

    @property
    def competing_classes(self) -> bool:
        """Return whether more than one class survived within this pass."""

        return len(self.class_hypotheses) > 1


@dataclass
class MCDOTargetCluster:
    """Class-agnostic spatial cluster with per-pass competing hypotheses."""

    cluster_id: int
    observations: dict[int, dict[tuple[int, str], Detection]] = field(
        default_factory=dict
    )

    def add(self, pass_index: int, detection: Detection) -> None:
        """Keep the strongest detection for each class in a pass."""

        hypotheses = self.observations.setdefault(pass_index, {})
        key = (detection.class_id, detection.class_name)
        current = hypotheses.get(key)
        if current is None or _stronger_detection(detection, current):
            hypotheses[key] = detection

    def detections_for_pass(self, pass_index: int) -> tuple[Detection, ...]:
        """Return retained hypotheses in deterministic class order."""

        return tuple(
            sorted(
                self.observations.get(pass_index, {}).values(),
                key=lambda item: (item.class_name, item.class_id, *item.bbox),
            )
        )

    def representative_box(self, pass_index: int) -> BoundingBox:
        """Return a confidence-weighted box independent of the winner class."""

        detections = self.detections_for_pass(pass_index)
        if not detections:
            raise ValueError(f"Cluster {self.cluster_id} has no pass {pass_index}")
        total = sum(detection.confidence for detection in detections)
        if total > 0.0:
            return tuple(
                sum(detection.bbox[axis] * detection.confidence for detection in detections)
                / total
                for axis in range(4)
            )  # type: ignore[return-value]
        return tuple(
            fmean(detection.bbox[axis] for detection in detections)
            for axis in range(4)
        )  # type: ignore[return-value]

    def reference_box(self) -> BoundingBox:
        """Return the mean representative box over detected passes."""

        if not self.observations:
            raise ValueError("Cannot calculate a reference for an empty V2 cluster")
        boxes = [self.representative_box(index) for index in sorted(self.observations)]
        return tuple(fmean(box[axis] for box in boxes) for axis in range(4))  # type: ignore[return-value]


@dataclass(frozen=True)
class MCDOTargetResult:
    """Transparent existence, class, confidence, and localization dimensions."""

    target_id: str
    sample_count: int
    detected_count: int
    persistence: float
    detected_pass_indices: tuple[int, ...]
    missing_pass_indices: tuple[int, ...]
    winner_class_distribution: Mapping[str, int]
    dominant_winner_class: str | None
    winner_class_agreement: float | None
    winner_class_entropy_bits: float | None
    competition_count: int
    competition_rate: float | None
    winner_confidence_mean: float | None
    winner_confidence_std: float | None
    class_evidence_share: Mapping[str, float] | None
    evidence_entropy_bits: float | None
    bbox_center_std_pixels: AxisVariation | None
    bbox_size_std_pixels: AxisVariation | None
    reference_bbox_xyxy: BoundingBox | None
    mean_reference_iou: float | None
    minimum_reference_iou: float | None
    existence_status: str
    classification_status: str
    localization_status: str
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready result using no ground-truth terminology."""

        return asdict(self)


@dataclass(frozen=True)
class MCDOFrameAnalysis:
    """V2 analysis for one exact frame across stochastic model passes."""

    method: str
    sample_count: int
    match_iou: float
    samples: tuple[tuple[Detection, ...], ...]
    clusters: tuple[MCDOTargetCluster, ...]
    targets: tuple[MCDOTargetResult, ...]

    @property
    def status(self) -> str:
        """Return a bounded summary while retaining all component statuses."""

        if not self.targets:
            return "NO STOCHASTIC DETECTIONS"
        if any(target.existence_status == "UNSTABLE / REVIEW" for target in self.targets):
            return "MODEL OUTPUT UNSTABLE / REVIEW"
        if any(target.localization_status == "UNSTABLE / REVIEW" for target in self.targets):
            return "LOCALIZATION UNSTABLE / REVIEW"
        if any(target.classification_status == "UNCERTAIN" for target in self.targets):
            return "CLASSIFICATION UNCERTAIN"
        if any(
            "VARIABLE" in (
                target.existence_status,
                target.classification_status,
                target.localization_status,
            )
            for target in self.targets
        ):
            return "MODEL OUTPUT VARIABLE / REVIEW"
        return "MODEL OUTPUT STABLE"

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready metadata without embedding pixels."""

        return {
            "method": self.method,
            "scientific_scope": (
                "approximate model/epistemic uncertainty from active internal dropout "
                "on one unchanged frame; not calibrated correctness probability"
            ),
            "sample_count": self.sample_count,
            "match_iou": self.match_iou,
            "detection_counts_by_pass": [len(sample) for sample in self.samples],
            "status": self.status,
            "targets": [target.to_dict() for target in self.targets],
            "undefined_when_no_detections": {
                "winner_class_agreement": None,
                "winner_class_entropy_bits": None,
                "class_evidence_share": None,
            }
            if not self.targets
            else None,
        }


class StochasticPassSession(Protocol):
    """Prepared lower-level forward path for one unchanged input tensor."""

    def detect_pass(self) -> Sequence[Detection]:
        """Run one stochastic forward/NMS pass."""


class MCDOPassRunner(Protocol):
    """Prepare the exact frame once and return a repeated-pass session."""

    def prepare(self, exact_frame: Image) -> StochasticPassSession:
        """Preprocess one defensive frame copy for repeated forwards."""


def _module_types(
    dropout_type: type[Any] | None,
    batchnorm_type: type[Any] | None,
    convolution_type: type[Any] | None,
) -> tuple[type[Any], type[Any], type[Any]]:
    if dropout_type and batchnorm_type and convolution_type:
        return dropout_type, batchnorm_type, convolution_type
    try:
        import torch.nn as nn
    except ImportError as error:
        raise MCDOV2Error("PyTorch is required for V2 architecture validation") from error
    return (
        dropout_type or nn.Dropout2d,
        batchnorm_type or nn.modules.batchnorm._BatchNorm,
        convolution_type or nn.Conv2d,
    )


def validate_mcdo_architecture(
    core_model: Any,
    config: MCDOV2Config | None = None,
    *,
    dropout_type: type[Any] | None = None,
    batchnorm_type: type[Any] | None = None,
    convolution_type: type[Any] | None = None,
) -> MCDOArchitectureReport:
    """Fail closed unless the checkpoint matches the notebook's six placements."""

    settings = config or MCDOV2Config()
    dropout_type, batchnorm_type, convolution_type = _module_types(
        dropout_type, batchnorm_type, convolution_type
    )
    try:
        modules = list(core_model.modules())
        head = core_model.model[-1]
    except (AttributeError, IndexError, TypeError) as error:
        raise MCDOV2Error("V2 checkpoint has no compatible YOLO detection head") from error

    dropouts = [module for module in modules if isinstance(module, dropout_type)]
    batchnorms = [module for module in modules if isinstance(module, batchnorm_type)]
    if len(dropouts) != settings.dropout_count:
        raise MCDOV2Error(
            f"Expected 6 Dropout2d layers, found {len(dropouts)}; V2 disabled"
        )
    probabilities = [float(getattr(module, "p", math.nan)) for module in dropouts]
    if any(
        not math.isclose(
            probability,
            settings.dropout_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for probability in probabilities
    ):
        raise MCDOV2Error(
            f"Every V2 Dropout2d must use p=0.20; found {probabilities}"
        )

    placements: list[str] = []
    placed_ids: set[int] = set()
    for branch_name in ("cv2", "cv3"):
        branches = getattr(head, branch_name, None)
        if branches is None or len(branches) != VALIDATED_DETECTION_SCALES:
            found = 0 if branches is None else len(branches)
            raise MCDOV2Error(
                f"Expected three {branch_name} detection scales, found {found}"
            )
        for scale_index, sequence in enumerate(branches):
            children = list(sequence.children())
            if len(children) < 2:
                raise MCDOV2Error(
                    f"{branch_name}[{scale_index}] cannot contain the validated placement"
                )
            dropout = children[-2]
            prediction_conv = children[-1]
            if not isinstance(dropout, dropout_type) or not isinstance(
                prediction_conv, convolution_type
            ):
                raise MCDOV2Error(
                    "Validated V2 requires Dropout2d immediately before the final "
                    f"prediction Conv2d in {branch_name}[{scale_index}]"
                )
            placed_ids.add(id(dropout))
            placements.append(f"{branch_name}[{scale_index}].before_final_conv")
    if placed_ids != {id(module) for module in dropouts}:
        raise MCDOV2Error("V2 contains Dropout2d outside the six validated placements")

    return MCDOArchitectureReport(
        dropout_count=len(dropouts),
        dropout_probability=settings.dropout_probability,
        batchnorm_count=len(batchnorms),
        placements=tuple(placements),
        all_dropout_active=all(bool(module.training) for module in dropouts),
        all_batchnorm_eval=not any(bool(module.training) for module in batchnorms),
        model_eval=not bool(getattr(core_model, "training", True)),
    )


def validate_mcdo_inference_state(
    core_model: Any,
    config: MCDOV2Config | None = None,
    *,
    dropout_type: type[Any] | None = None,
    batchnorm_type: type[Any] | None = None,
    convolution_type: type[Any] | None = None,
) -> MCDOArchitectureReport:
    """Verify active Dropout2d and evaluation-mode BatchNorm/model state."""

    report = validate_mcdo_architecture(
        core_model,
        config,
        dropout_type=dropout_type,
        batchnorm_type=batchnorm_type,
        convolution_type=convolution_type,
    )
    if not report.model_eval:
        raise MCDOV2Error("V2 core model must remain globally in evaluation mode")
    if not report.all_dropout_active:
        raise MCDOV2Error("All six V2 Dropout2d layers must be active")
    if not report.all_batchnorm_eval:
        raise MCDOV2Error("Every BatchNorm layer must remain in evaluation mode")
    return report


def activate_mcdo_inference(
    core_model: Any,
    config: MCDOV2Config | None = None,
    *,
    dropout_type: type[Any] | None = None,
    batchnorm_type: type[Any] | None = None,
    convolution_type: type[Any] | None = None,
) -> MCDOArchitectureReport:
    """Set global eval, then activate only the six validated Dropout2d layers."""

    settings = config or MCDOV2Config()
    dropout_type, batchnorm_type, convolution_type = _module_types(
        dropout_type, batchnorm_type, convolution_type
    )
    validate_mcdo_architecture(
        core_model,
        settings,
        dropout_type=dropout_type,
        batchnorm_type=batchnorm_type,
        convolution_type=convolution_type,
    )
    try:
        core_model.eval()
        for module in core_model.modules():
            if isinstance(module, dropout_type):
                module.train(True)
            elif isinstance(module, batchnorm_type):
                module.eval()
    except (AttributeError, TypeError) as error:
        raise MCDOV2Error("V2 model does not support PyTorch inference state") from error
    return validate_mcdo_inference_state(
        core_model,
        settings,
        dropout_type=dropout_type,
        batchnorm_type=batchnorm_type,
        convolution_type=convolution_type,
    )


def _stronger_detection(candidate: Detection, current: Detection) -> bool:
    return (-candidate.confidence, *candidate.bbox) < (
        -current.confidence,
        *current.bbox,
    )


def _detection_key(detection: Detection) -> tuple[object, ...]:
    return (*detection.bbox, detection.class_name, detection.class_id, -detection.confidence)


def cluster_mcdo_samples(
    samples: Sequence[Sequence[Detection]], match_iou: float = 0.50
) -> list[MCDOTargetCluster]:
    """Cluster spatially first while retaining same-pass competing classes."""

    if not 0.0 < match_iou <= 1.0:
        raise ValueError("match_iou must be greater than 0 and at most 1")
    clusters: list[MCDOTargetCluster] = []
    next_cluster_id = 1
    for pass_index, sample in enumerate(samples):
        for detection in sorted(sample, key=_detection_key):
            candidates = [
                (
                    intersection_over_union(cluster.reference_box(), detection.bbox),
                    cluster,
                )
                for cluster in clusters
            ]
            eligible = [item for item in candidates if item[0] >= match_iou]
            if eligible:
                _, selected = min(
                    eligible,
                    key=lambda item: (-item[0], item[1].cluster_id),
                )
            else:
                selected = MCDOTargetCluster(next_cluster_id)
                clusters.append(selected)
                next_cluster_id += 1
            selected.add(pass_index, detection)
    return clusters


def _entropy(shares: Sequence[float]) -> float:
    value = -sum(share * math.log2(share) for share in shares if share > 0.0)
    return 0.0 if value == 0.0 else value


def _std(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _pass_record(cluster: MCDOTargetCluster, pass_index: int) -> MCDOPassRecord:
    hypotheses = cluster.detections_for_pass(pass_index)
    if not hypotheses:
        raise ValueError("Cannot summarize an empty stochastic pass")
    winner = min(
        hypotheses,
        key=lambda item: (-item.confidence, item.class_name, item.class_id, *item.bbox),
    )
    confidence_sum = sum(item.confidence for item in hypotheses)
    evidence: Mapping[str, float] | None = None
    if confidence_sum > 0.0:
        evidence = {
            item.class_name: item.confidence / confidence_sum
            for item in hypotheses
        }
    return MCDOPassRecord(
        pass_index=pass_index,
        winner_class_id=winner.class_id,
        winner_class_name=winner.class_name,
        winner_confidence=winner.confidence,
        class_hypotheses=hypotheses,
        evidence_share=evidence,
        representative_bbox_xyxy=cluster.representative_box(pass_index),
    )


def existence_status(persistence: float) -> str:
    """Classify detection persistence without implying correctness."""

    if persistence >= 0.95:
        return "STABLE"
    if persistence >= 0.75:
        return "VARIABLE"
    return "UNSTABLE / REVIEW"


def classification_status(
    agreement: float | None,
    competition_rate: float | None,
    evidence_entropy_bits: float | None,
) -> str:
    """Classify across/within-pass class behavior using visible dimensions."""

    if agreement is None or competition_rate is None:
        return "N/A"
    if (
        agreement < 0.75
        or competition_rate >= 0.50
        or (evidence_entropy_bits is not None and evidence_entropy_bits >= 0.80)
    ):
        return "UNCERTAIN"
    if (
        agreement < 0.95
        or competition_rate > 0.0
        or (evidence_entropy_bits is not None and evidence_entropy_bits >= 0.30)
    ):
        return "VARIABLE"
    return "STABLE"


def localization_status(mean_reference_iou: float | None) -> str:
    """Classify predicted-box consistency against the mean predicted box."""

    if mean_reference_iou is None:
        return "N/A"
    if mean_reference_iou >= 0.80:
        return "STABLE"
    if mean_reference_iou >= 0.60:
        return "VARIABLE"
    return "UNSTABLE / REVIEW"


def _interpret_dimensions(existence: str, classification: str, localization: str) -> str:
    if classification == "UNCERTAIN" and existence == localization == "STABLE":
        return "Existence stable; localization stable; classification uncertain"
    if existence == classification == localization == "STABLE":
        return "Model output stable across the stochastic passes"
    if "UNSTABLE / REVIEW" in (existence, localization):
        return "Model output unstable; review existence and localization dimensions"
    if classification == "UNCERTAIN":
        return "Classification uncertain; review competing model evidence"
    if classification == "N/A":
        return "No stochastic detections; class and localization statistics are N/A"
    return "Model output variable across the stochastic passes"


def calculate_mcdo_target(
    cluster: MCDOTargetCluster, sample_count: int
) -> MCDOTargetResult:
    """Calculate notebook-derived V2 metrics for one physical-object cluster."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    invalid = [
        index for index in cluster.observations if index < 0 or index >= sample_count
    ]
    if invalid:
        raise ValueError(f"Observation indices outside pass range: {invalid}")
    detected_indices = tuple(sorted(cluster.observations))
    detected_count = len(detected_indices)
    persistence = detected_count / sample_count
    existence = existence_status(persistence)
    if detected_count == 0:
        classification = "N/A"
        localization = "N/A"
        return MCDOTargetResult(
            target_id=f"target_{cluster.cluster_id}",
            sample_count=sample_count,
            detected_count=0,
            persistence=0.0,
            detected_pass_indices=(),
            missing_pass_indices=tuple(range(sample_count)),
            winner_class_distribution={},
            dominant_winner_class=None,
            winner_class_agreement=None,
            winner_class_entropy_bits=None,
            competition_count=0,
            competition_rate=None,
            winner_confidence_mean=None,
            winner_confidence_std=None,
            class_evidence_share=None,
            evidence_entropy_bits=None,
            bbox_center_std_pixels=None,
            bbox_size_std_pixels=None,
            reference_bbox_xyxy=None,
            mean_reference_iou=None,
            minimum_reference_iou=None,
            existence_status=existence,
            classification_status=classification,
            localization_status=localization,
            interpretation=_interpret_dimensions(existence, classification, localization),
        )

    records = [_pass_record(cluster, index) for index in detected_indices]
    winner_counts = Counter(record.winner_class_name for record in records)
    sorted_counts = sorted(winner_counts.items(), key=lambda item: (-item[1], item[0]))
    dominant_class, dominant_count = sorted_counts[0]
    winner_distribution = dict(sorted(winner_counts.items()))
    winner_agreement = dominant_count / detected_count
    winner_entropy = _entropy(
        [count / detected_count for count in winner_distribution.values()]
    )
    competition_count = sum(record.competing_classes for record in records)
    competition_rate = competition_count / detected_count

    evidence_records = [record for record in records if record.evidence_share is not None]
    evidence_share: Mapping[str, float] | None = None
    evidence_entropy: float | None = None
    if evidence_records:
        class_names = sorted(
            {
                class_name
                for record in evidence_records
                for class_name in (record.evidence_share or {})
            }
        )
        evidence_share = {
            class_name: fmean(
                (record.evidence_share or {}).get(class_name, 0.0)
                for record in evidence_records
            )
            for class_name in class_names
        }
        evidence_entropy = _entropy(list(evidence_share.values()))

    boxes = [record.representative_bbox_xyxy for record in records]
    reference = tuple(fmean(box[axis] for box in boxes) for axis in range(4))  # type: ignore[assignment]
    reference_ious = [intersection_over_union(reference, box) for box in boxes]
    centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in boxes]
    sizes = [(box[2] - box[0], box[3] - box[1]) for box in boxes]
    classification = classification_status(
        winner_agreement, competition_rate, evidence_entropy
    )
    mean_reference_iou = fmean(reference_ious)
    localization = localization_status(mean_reference_iou)
    winner_confidences = [record.winner_confidence for record in records]
    return MCDOTargetResult(
        target_id=f"target_{cluster.cluster_id}",
        sample_count=sample_count,
        detected_count=detected_count,
        persistence=persistence,
        detected_pass_indices=detected_indices,
        missing_pass_indices=tuple(
            index for index in range(sample_count) if index not in cluster.observations
        ),
        winner_class_distribution=winner_distribution,
        dominant_winner_class=dominant_class,
        winner_class_agreement=winner_agreement,
        winner_class_entropy_bits=winner_entropy,
        competition_count=competition_count,
        competition_rate=competition_rate,
        winner_confidence_mean=fmean(winner_confidences),
        winner_confidence_std=_std(winner_confidences),
        class_evidence_share=evidence_share,
        evidence_entropy_bits=evidence_entropy,
        bbox_center_std_pixels=AxisVariation(
            _std([center[0] for center in centers]),
            _std([center[1] for center in centers]),
        ),
        bbox_size_std_pixels=AxisVariation(
            _std([size[0] for size in sizes]),
            _std([size[1] for size in sizes]),
        ),
        reference_bbox_xyxy=reference,
        mean_reference_iou=mean_reference_iou,
        minimum_reference_iou=min(reference_ious),
        existence_status=existence,
        classification_status=classification,
        localization_status=localization,
        interpretation=_interpret_dimensions(existence, classification, localization),
    )


def _validate_image(image: Image) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("V2 exact_frame must be a NumPy array")
    if image.dtype != np.uint8 or image.size == 0:
        raise ValueError("V2 exact_frame must be a non-empty uint8 image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("V2 exact_frame must be an OpenCV BGR image")


def run_mcdo_frame(
    exact_frame: Image,
    runner: MCDOPassRunner,
    config: MCDOV2Config | None = None,
) -> MCDOFrameAnalysis:
    """Run repeated stochastic forwards on one defensive, unchanged frame copy."""

    settings = config or MCDOV2Config()
    _validate_image(exact_frame)
    frozen = exact_frame.copy()
    expected_pixels = frozen.copy()
    session = runner.prepare(frozen)
    if not np.array_equal(frozen, expected_pixels):
        raise MCDOV2Error("V2 preprocessing modified the exact frozen frame")
    samples: list[tuple[Detection, ...]] = []
    for _ in range(settings.sample_count):
        detections = tuple(session.detect_pass())
        if not all(isinstance(item, Detection) for item in detections):
            raise TypeError("V2 pass runner returned an unsupported detection type")
        if not np.array_equal(frozen, expected_pixels):
            raise MCDOV2Error("V2 pass runner modified the exact frozen frame")
        samples.append(detections)
    clusters = cluster_mcdo_samples(samples, settings.match_iou)
    targets = tuple(
        calculate_mcdo_target(cluster, settings.sample_count)
        for cluster in sorted(clusters, key=lambda item: item.cluster_id)
    )
    return MCDOFrameAnalysis(
        method=METHOD_ID,
        sample_count=settings.sample_count,
        match_iou=settings.match_iou,
        samples=tuple(samples),
        clusters=tuple(clusters),
        targets=targets,
    )
