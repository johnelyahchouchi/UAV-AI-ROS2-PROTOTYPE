"""Trusted Ultralytics 8.4.107 lower-level forward path for MC Dropout V2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .detection import Detection
from .mc_dropout_v2 import (
    MCDOArchitectureReport,
    MCDOV2Config,
    MCDOV2Error,
    activate_mcdo_inference,
    validate_mcdo_inference_state,
)


VALIDATED_ULTRALYTICS_VERSION = "8.4.107"


def load_verified_mcdo_checkpoint(
    model_path: Path,
    registry_path: Path | None = None,
    *,
    verifier: Callable[..., str] | None = None,
    loader: Callable[..., Any] | None = None,
) -> tuple[str, Any]:
    """Verify the checkpoint hash before any PyTorch/YOLO deserialization."""

    if verifier is None or loader is None:
        from uav_security.model_integrity import load_trusted_yolo, verify_trusted_model

        verifier = verifier or verify_trusted_model
        loader = loader or load_trusted_yolo
    digest = verifier(model_path, registry_path)
    try:
        model = loader(model_path, registry_path=registry_path)
    except Exception as error:
        raise MCDOV2Error(f"Trusted V2 checkpoint could not be loaded: {error}") from error
    return digest, model


def _class_name(names: Any, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


class _UltralyticsMCDOSession:
    """Prepared same-tensor session that performs one raw forward plus NMS."""

    def __init__(
        self,
        owner: "TrustedUltralyticsMCDORunner",
        tensor: Any,
        processed_shape: tuple[int, int],
        original_shape: tuple[int, int],
    ) -> None:
        self.owner = owner
        self.tensor = tensor
        self.processed_shape = processed_shape
        self.original_shape = original_shape

    def detect_pass(self) -> list[Detection]:
        """Run one stochastic core-model forward without high-level ``predict``."""

        owner = self.owner
        validate_mcdo_inference_state(owner.core_model, owner.config)
        try:
            with owner.torch.no_grad():
                raw = owner.core_model(self.tensor)
            if isinstance(raw, tuple):
                raw = raw[0]
            post_nms = owner.non_max_suppression(
                raw,
                conf_thres=owner.config.confidence,
                iou_thres=owner.config.nms_iou,
                nc=owner.class_count,
            )
        except Exception as error:
            raise MCDOV2Error(f"V2 lower-level forward/NMS failed: {error}") from error
        if not post_nms:
            return []
        rows = post_nms[0]
        if rows is None or len(rows) == 0:
            return []
        boxes = rows[:, :4].clone()
        owner.scale_boxes(self.processed_shape, boxes, self.original_shape)
        height, width = self.original_shape
        output: list[Detection] = []
        for row, scaled_box in zip(rows, boxes):
            try:
                values = scaled_box.detach().cpu().tolist()
                confidence = float(row[4].detach().cpu().item())
                class_id = int(row[5].detach().cpu().item())
                x1 = max(0.0, min(float(width - 1), float(values[0])))
                y1 = max(0.0, min(float(height - 1), float(values[1])))
                x2 = max(0.0, min(float(width), float(values[2])))
                y2 = max(0.0, min(float(height), float(values[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                output.append(
                    Detection(
                        class_id=class_id,
                        class_name=_class_name(owner.names, class_id),
                        confidence=max(0.0, min(1.0, confidence)),
                        bbox=(x1, y1, x2, y2),
                    )
                )
            except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
                continue
        return output


class TrustedUltralyticsMCDORunner:
    """Secure V2 checkpoint loader and notebook-validated inference adapter."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        config: MCDOV2Config | None = None,
        registry_path: Path | None = None,
        verifier: Callable[..., str] | None = None,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        self.config = config or MCDOV2Config()
        self.model_sha256, self.yolo = load_verified_mcdo_checkpoint(
            self.model_path,
            registry_path,
            verifier=verifier,
            loader=loader,
        )
        try:
            import cv2
            import torch
            import ultralytics
            from ultralytics.data.augment import LetterBox
            from ultralytics.utils.nms import non_max_suppression
            from ultralytics.utils.ops import scale_boxes
        except ImportError as error:
            raise MCDOV2Error(
                "V2 requires the controlled Ultralytics/PyTorch/OpenCV environment"
            ) from error
        if ultralytics.__version__ != VALIDATED_ULTRALYTICS_VERSION:
            raise MCDOV2Error(
                "Validated V2 requires ultralytics==8.4.107; found "
                f"{ultralytics.__version__}"
            )
        self.cv2 = cv2
        self.torch = torch
        self.letterbox_factory = LetterBox
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.device = "cpu" if device == "cpu" else f"cuda:{device}"
        if self.device.startswith("cuda:") and not torch.cuda.is_available():
            raise MCDOV2Error("V2 requested CUDA but PyTorch reports it unavailable")
        try:
            self.core_model = self.yolo.model.to(self.device)
        except Exception as error:
            raise MCDOV2Error(f"V2 model could not move to {self.device}: {error}") from error
        self.names = getattr(self.yolo, "names", None) or getattr(
            self.core_model, "names", {}
        )
        self.class_count = len(self.names)
        if self.class_count <= 0:
            raise MCDOV2Error("V2 checkpoint exposes no detection class names")
        self.architecture_report: MCDOArchitectureReport = activate_mcdo_inference(
            self.core_model, self.config
        )

    def prepare(self, exact_frame: Any) -> _UltralyticsMCDOSession:
        """Preprocess one BGR frame once and reuse the exact tensor for all passes."""

        activate_mcdo_inference(self.core_model, self.config)
        original = exact_frame.copy()
        try:
            letterbox = self.letterbox_factory(
                new_shape=(self.config.image_size, self.config.image_size),
                auto=False,
                stride=32,
            )
            processed = letterbox(image=original)
            processed = self.cv2.cvtColor(processed, self.cv2.COLOR_BGR2RGB)
            tensor = (
                self.torch.from_numpy(processed)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self.device)
                / 255.0
            ).contiguous()
        except Exception as error:
            raise MCDOV2Error(f"V2 frame preprocessing failed: {error}") from error
        return _UltralyticsMCDOSession(
            self,
            tensor,
            (int(processed.shape[0]), int(processed.shape[1])),
            (int(exact_frame.shape[0]), int(exact_frame.shape[1])),
        )
