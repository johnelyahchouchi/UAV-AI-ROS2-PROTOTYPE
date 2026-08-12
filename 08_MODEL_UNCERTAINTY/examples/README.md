# Example usage

This subsystem intentionally does not include weights or datasets. Supply an existing
local Ultralytics detection checkpoint and a local image.

From the repository root in PowerShell:

    $env:PYTHONPATH = Join-Path $PWD "08_MODEL_UNCERTAINTY\src"
    & $env:UAV_YOLO_PYTHON -m uav_uncertainty.mc_stability_runner `
      --model "C:\path\to\model.pt" `
      --image "C:\path\to\uav_frame.jpg" `
      --samples 10 `
      --seed 42 `
      --imgsz 960 `
      --conf 0.25 `
      --iou 0.45 `
      --match-iou 0.50 `
      --device 0

Illustrative console shape (values are not claimed results):

    Target 1
    Dominant class: military_vehicle
    Detected: 10/11
    Persistence: 0.909
    Mean confidence: 0.780
    Confidence std: 0.050
    Class agreement: 0.900
    Class entropy (bits): 0.469
    Center std (px): x=1.420, y=1.110
    Size std (px): w=2.030, h=1.770
    Mean IoU to reference: 0.930

`--samples N` means N perturbed samples and N + 1 total inference samples because one
clean baseline prediction is always included. Therefore the denominator is 11 when
`--samples 10`. Inspect summary.json for the exact perturbation parameters and raw
per-target fields. No reliability category or combined uncertainty score is produced.

An existing summary.json or targets.csv is protected by default. Pass --overwrite only
when you intentionally want to atomically replace those two result files; unrelated
files in the run directory are preserved.
