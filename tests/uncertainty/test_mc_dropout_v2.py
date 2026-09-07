from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from uav_uncertainty.detection import Detection
from uav_uncertainty.mc_dropout_ultralytics import load_verified_mcdo_checkpoint
from uav_uncertainty.mc_dropout_v2 import (
    MCDOTargetCluster,
    MCDOV2Config,
    MCDOV2Error,
    activate_mcdo_inference,
    calculate_mcdo_target,
    cluster_mcdo_samples,
    run_mcdo_frame,
    validate_mcdo_architecture,
)


class FakeModule:
    def __init__(self, *children: "FakeModule") -> None:
        self.training = True
        self._children = list(children)

    def children(self):
        return iter(self._children)

    def modules(self):
        yield self
        for child in self._children:
            yield from child.modules()

    def train(self, mode: bool = True):
        self.training = mode
        for child in self._children:
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)


class FakeDropout2d(FakeModule):
    def __init__(self, p: float = 0.20) -> None:
        super().__init__()
        self.p = p


class FakeBatchNorm(FakeModule):
    pass


class FakeConv2d(FakeModule):
    pass


class FakeSequence(FakeModule):
    pass


class FakeHead(FakeModule):
    def __init__(self, probability: float = 0.20) -> None:
        self.cv2 = [
            FakeSequence(FakeBatchNorm(), FakeDropout2d(probability), FakeConv2d())
            for _ in range(3)
        ]
        self.cv3 = [
            FakeSequence(FakeBatchNorm(), FakeDropout2d(probability), FakeConv2d())
            for _ in range(3)
        ]
        super().__init__(*self.cv2, *self.cv3)


class FakeCoreModel(FakeModule):
    def __init__(self, probability: float = 0.20) -> None:
        self.body = FakeModule(FakeBatchNorm())
        self.head = FakeHead(probability)
        self.model = [self.body, self.head]
        super().__init__(self.body, self.head)


MODULE_TYPES = {
    "dropout_type": FakeDropout2d,
    "batchnorm_type": FakeBatchNorm,
    "convolution_type": FakeConv2d,
}


def detection(
    class_id: int = 0,
    class_name: str = "military_tank",
    confidence: float = 0.8,
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 30.0, 30.0),
) -> Detection:
    return Detection(class_id, class_name, confidence, bbox)


class SequenceSession:
    def __init__(self, runner: "SequenceRunner") -> None:
        self.runner = runner

    def detect_pass(self):
        index = self.runner.pass_calls
        self.runner.pass_calls += 1
        return self.runner.outputs[index % len(self.runner.outputs)]


class SequenceRunner:
    def __init__(self, outputs: list[list[Detection]]) -> None:
        self.outputs = outputs
        self.prepare_calls = 0
        self.pass_calls = 0
        self.prepared_pixels: np.ndarray | None = None

    def prepare(self, exact_frame: np.ndarray) -> SequenceSession:
        self.prepare_calls += 1
        self.prepared_pixels = exact_frame.copy()
        return SequenceSession(self)


def test_activation_keeps_model_and_batchnorm_eval_with_only_dropout_active() -> None:
    model = FakeCoreModel()

    report = activate_mcdo_inference(model, **MODULE_TYPES)

    modules = list(model.modules())
    assert model.training is False
    assert report.model_eval is True
    assert report.dropout_count == 6
    assert report.dropout_probability == pytest.approx(0.20)
    assert report.placements == tuple(
        f"{branch}[{scale}].before_final_conv"
        for branch in ("cv2", "cv3")
        for scale in range(3)
    )
    assert all(module.training for module in modules if isinstance(module, FakeDropout2d))
    assert not any(module.training for module in modules if isinstance(module, FakeBatchNorm))
    assert not any(
        module.training
        for module in modules
        if not isinstance(module, FakeDropout2d)
    )


def test_wrong_dropout_count_is_rejected() -> None:
    model = FakeCoreModel()
    model.head.cv3[2]._children[-2] = FakeModule()

    with pytest.raises(MCDOV2Error, match="Expected 6 Dropout2d layers, found 5"):
        validate_mcdo_architecture(model, **MODULE_TYPES)


def test_wrong_dropout_probability_is_rejected() -> None:
    model = FakeCoreModel(probability=0.30)

    with pytest.raises(MCDOV2Error, match="p=0.20"):
        validate_mcdo_architecture(model, **MODULE_TYPES)


def test_wrong_dropout_placement_is_rejected() -> None:
    model = FakeCoreModel()
    branch = model.head.cv2[0]
    branch._children[-2], branch._children[-1] = (
        branch._children[-1],
        branch._children[-2],
    )

    with pytest.raises(MCDOV2Error, match="immediately before"):
        validate_mcdo_architecture(model, **MODULE_TYPES)


@pytest.mark.parametrize(
    "kwargs",
    ({"dropout_count": 5}, {"dropout_probability": 0.30}),
)
def test_config_cannot_redefine_validated_architecture(kwargs) -> None:
    with pytest.raises(ValueError, match="Validated V2 requires"):
        MCDOV2Config(**kwargs)


def test_default_run_prepares_once_and_executes_twenty_same_frame_passes() -> None:
    original = np.full((18, 24, 3), 91, dtype=np.uint8)
    expected = original.copy()
    runner = SequenceRunner([[detection()]])

    result = run_mcdo_frame(original, runner)

    assert runner.prepare_calls == 1
    assert runner.pass_calls == 20
    assert np.array_equal(original, expected)
    assert np.array_equal(runner.prepared_pixels, expected)
    assert result.sample_count == 20
    assert len(result.samples) == 20


def test_deterministic_fake_outputs_have_zero_disagreement() -> None:
    result = run_mcdo_frame(
        np.zeros((32, 32, 3), dtype=np.uint8),
        SequenceRunner([[detection()]]),
    )
    target = result.targets[0]

    assert target.persistence == pytest.approx(1.0)
    assert target.winner_class_agreement == pytest.approx(1.0)
    assert target.winner_class_entropy_bits == pytest.approx(0.0)
    assert target.competition_rate == pytest.approx(0.0)
    assert target.evidence_entropy_bits == pytest.approx(0.0)
    assert target.winner_confidence_std == pytest.approx(0.0)
    assert target.bbox_center_std_pixels.x == pytest.approx(0.0)
    assert target.bbox_center_std_pixels.y == pytest.approx(0.0)
    assert target.bbox_size_std_pixels.x == pytest.approx(0.0)
    assert target.bbox_size_std_pixels.y == pytest.approx(0.0)
    assert target.mean_reference_iou == pytest.approx(1.0)
    assert target.minimum_reference_iou == pytest.approx(1.0)
    assert result.status == "MODEL OUTPUT STABLE"


def test_class_agnostic_matching_preserves_class_changes_in_one_cluster() -> None:
    samples = [
        [detection(0, "military_tank", 0.8)],
        [detection(1, "military_vehicle", 0.7)],
    ]

    clusters = cluster_mcdo_samples(samples)
    target = calculate_mcdo_target(clusters[0], 2)

    assert len(clusters) == 1
    assert target.winner_class_distribution == {
        "military_tank": 1,
        "military_vehicle": 1,
    }
    assert target.winner_class_agreement == pytest.approx(0.5)
    assert target.winner_class_entropy_bits == pytest.approx(1.0)


def test_within_pass_competition_keeps_strongest_per_class_and_confidence_winner() -> None:
    tank_box = (10.0, 10.0, 30.0, 30.0)
    vehicle_box = (11.0, 11.0, 31.0, 31.0)
    samples = [
        [
            detection(0, "military_tank", 0.80, tank_box),
            detection(0, "military_tank", 0.55, (12.0, 12.0, 30.0, 30.0)),
            detection(1, "military_vehicle", 0.90, vehicle_box),
        ],
        [
            detection(0, "military_tank", 0.85, tank_box),
            detection(1, "military_vehicle", 0.70, vehicle_box),
        ],
        [detection(0, "military_tank", 0.50, tank_box)],
        [detection(1, "military_vehicle", 0.60, vehicle_box)],
    ]

    cluster = cluster_mcdo_samples(samples)[0]
    target = calculate_mcdo_target(cluster, 4)

    assert len(cluster.detections_for_pass(0)) == 2
    assert {item.confidence for item in cluster.detections_for_pass(0)} == {0.8, 0.9}
    assert target.winner_class_distribution == {
        "military_tank": 2,
        "military_vehicle": 2,
    }
    assert target.dominant_winner_class == "military_tank"  # deterministic tie break
    assert target.winner_class_agreement == pytest.approx(0.5)
    assert target.winner_class_entropy_bits == pytest.approx(1.0)
    assert target.competition_count == 2
    assert target.competition_rate == pytest.approx(0.5)
    assert target.winner_confidence_mean == pytest.approx((0.9 + 0.85 + 0.5 + 0.6) / 4)
    assert target.winner_confidence_std == pytest.approx(
        np.std([0.9, 0.85, 0.5, 0.6])
    )
    assert target.class_evidence_share is not None
    assert sum(target.class_evidence_share.values()) == pytest.approx(1.0)
    assert target.class_evidence_share["military_tank"] == pytest.approx(
        (0.8 / 1.7 + 0.85 / 1.55 + 1.0 + 0.0) / 4
    )
    assert target.evidence_entropy_bits is not None
    assert target.evidence_entropy_bits > 0.99
    assert target.classification_status == "UNCERTAIN"


def test_stochastic_box_outputs_report_center_width_and_height_variation() -> None:
    cluster = cluster_mcdo_samples(
        [
            [detection(confidence=0.6, bbox=(0.0, 0.0, 10.0, 10.0))],
            [detection(confidence=0.8, bbox=(2.0, 4.0, 14.0, 16.0))],
        ],
        match_iou=0.20,
    )[0]

    target = calculate_mcdo_target(cluster, 2)

    assert target.winner_confidence_mean == pytest.approx(0.7)
    assert target.winner_confidence_std == pytest.approx(0.1)
    assert target.bbox_center_std_pixels.x == pytest.approx(1.5)
    assert target.bbox_center_std_pixels.y == pytest.approx(2.5)
    assert target.bbox_size_std_pixels.x == pytest.approx(1.0)
    assert target.bbox_size_std_pixels.y == pytest.approx(1.0)
    assert 0.0 < target.minimum_reference_iou < target.mean_reference_iou < 1.0


def test_zero_detection_statistics_are_na_instead_of_zero() -> None:
    target = calculate_mcdo_target(MCDOTargetCluster(cluster_id=1), 20)

    assert target.detected_count == 0
    assert target.winner_class_agreement is None
    assert target.winner_class_entropy_bits is None
    assert target.class_evidence_share is None
    assert target.evidence_entropy_bits is None
    assert target.winner_confidence_mean is None
    assert target.reference_bbox_xyxy is None
    assert target.localization_status == "N/A"


def test_multiple_physical_objects_form_separate_clusters() -> None:
    samples = [
        [
            detection(bbox=(0.0, 0.0, 20.0, 20.0)),
            detection(1, "military_vehicle", 0.7, (100.0, 100.0, 130.0, 130.0)),
        ],
        [
            detection(bbox=(1.0, 1.0, 21.0, 21.0)),
            detection(1, "military_vehicle", 0.75, (101.0, 100.0, 131.0, 130.0)),
        ],
    ]

    clusters = cluster_mcdo_samples(samples)

    assert len(clusters) == 2
    assert [len(cluster.observations) for cluster in clusters] == [2, 2]


def test_runner_mutation_of_frozen_frame_fails_closed() -> None:
    class MutatingRunner:
        def prepare(self, exact_frame):
            exact_frame[0, 0, 0] = 255
            return SequenceSession(SequenceRunner([[]]))

    with pytest.raises(MCDOV2Error, match="preprocessing modified"):
        run_mcdo_frame(
            np.zeros((4, 4, 3), dtype=np.uint8),
            MutatingRunner(),
            MCDOV2Config(sample_count=1),
        )


def test_live_result_metadata_uses_reference_not_ground_truth_terms() -> None:
    result = run_mcdo_frame(
        np.zeros((32, 32, 3), dtype=np.uint8),
        SequenceRunner([[detection()]]),
        MCDOV2Config(sample_count=2),
    )
    serialized = json.dumps(result.to_dict(), sort_keys=True).lower()

    assert "ground truth" not in serialized
    assert "ground_truth" not in serialized
    assert "gt_iou" not in serialized
    assert "reference_iou" in serialized


def test_checkpoint_verification_occurs_before_loader(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v2.pt"
    checkpoint.write_bytes(b"not deserialized")
    calls: list[str] = []

    digest, model = load_verified_mcdo_checkpoint(
        checkpoint,
        verifier=lambda path, registry: calls.append("verify") or "a" * 64,
        loader=lambda path, *, registry_path: calls.append("load") or object(),
    )

    assert calls == ["verify", "load"]
    assert digest == "a" * 64
    assert model is not None


def test_failed_checkpoint_verification_never_calls_loader(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v2.pt"
    checkpoint.touch()
    loaded = False

    def reject(path, registry):
        raise RuntimeError("untrusted")

    def loader(path, *, registry_path):
        nonlocal loaded
        loaded = True

    with pytest.raises(RuntimeError, match="untrusted"):
        load_verified_mcdo_checkpoint(checkpoint, verifier=reject, loader=loader)
    assert loaded is False


def test_ultralytics_adapter_uses_lower_level_forward_not_high_level_predict() -> None:
    from uav_uncertainty import mc_dropout_ultralytics

    source = inspect.getsource(mc_dropout_ultralytics._UltralyticsMCDOSession.detect_pass)
    assert "core_model(self.tensor)" in source
    assert ".predict(" not in source
