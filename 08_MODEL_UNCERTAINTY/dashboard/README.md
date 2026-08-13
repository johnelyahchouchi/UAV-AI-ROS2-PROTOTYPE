# UAV Model Uncertainty Dashboard

This local Gradio dashboard runs and inspects the existing Input Perturbation V1
stability evaluator. It supports images, bounded video-frame sampling, saved runs,
target and perturbation inspection, and comparison between experiment configurations.
It does not modify a model, control a UAV, use ROS 2, or require an internet connection.

## Launch

The launcher uses the existing dedicated YOLO environment. It does not fall back to a
different Python installation and does not install or upgrade detector packages.

From PowerShell:

```powershell
.\08_MODEL_UNCERTAINTY\dashboard\launch_uncertainty_dashboard.ps1
```

The launcher verifies Python, Gradio, NumPy, Matplotlib, OpenCV, Ultralytics, PyTorch,
Torchvision, CUDA, and the GPU name before opening:

```text
http://127.0.0.1:7861
```

Set `UAV_UNCERTAINTY_DASHBOARD_PORT` to use another port. The server always binds to
`127.0.0.1`, uses `share=False`, and does not expose a public Gradio link.

## Image experiments

1. Choose **Image** and upload a source image.
2. Select an existing local detection-model `.pt` file.
3. Choose Auto, GPU 0, or CPU and the detector settings.
4. Set the perturbed sample count and seed.
5. Select **Start Experiment**.

`N` means N perturbed images. Every image experiment also includes one clean baseline,
so the detector runs `N + 1` times. The method selector contains only the implemented
**Input Perturbation V1** method. Monte Carlo Dropout is not presented as available.

Progress reports model loading, clean inference, every perturbation sample, matching,
metric calculation, dashboard rendering, and output publication. Cancellation is
cooperative: the current detector call finishes, then partial staging data is removed.

## Reading the results

The dashboard deliberately keeps the raw indicators separate:

- **Detection persistence** is detected observations divided by all clean and perturbed
  samples. Higher values indicate more consistent detection in this experiment.
- **Confidence standard deviation** is the population standard deviation over detected
  observations. Lower values indicate less confidence variation.
- **Class agreement** is the dominant class count divided by detected observations.
- **Class entropy** is Shannon entropy in bits. Lower values indicate less class
  confusion within the matched target cluster.
- **Mean IoU** compares observed boxes with the clean reference box when present, or the
  mean observed box otherwise. Higher values indicate more stable localization.

These are stability and robustness measurements, not calibrated correctness
probabilities. Without labeled ground truth, the dashboard does not measure true
detection accuracy.

The Target Analysis tab shows every raw metric, the clean or selected perturbed image,
matched target IDs, class, confidence, missing samples, and the reference box source.
Selecting a target emphasizes its box in the sample viewer.

The overlap diagnostic reports pairs of reference boxes above the configured IoU
threshold. `Possible overlapping/alternative detection` is a review prompt only. It
does not merge clusters or assert that two detections are the same physical object.

## Perturbation analysis

The sample table shows the exact family and parameters, detections, and present/missing
target IDs. The family table summarizes observed counts, presence, misses, and confidence
change from the clean observation where a clean observation exists. Its statements are
factual, for example, `target_3 appeared in 2/2 jpeg_compression samples`; they are not
causal claims.

## Video experiments

A video is not treated as one Monte Carlo sample. The dashboard first chooses a bounded
set of frames, then runs a complete independent V1 image analysis on each selected frame.

- **Interval** starts at 0 seconds and samples every configured number of seconds, up to
  the maximum frame count.
- **Manual timestamps** accepts comma-separated seconds such as `5, 10, 15, 20, 30`.

Use **Calculate video workload** before starting. It reports selected frames,
perturbations per frame, `N + 1` inference calls per frame, and total detector calls.
Defaults are every 5 seconds and at most 20 frames; the dashboard never automatically
processes every video frame.

The Video Analysis tab shows one row per sampled frame and a timeline. Open a frame to
reuse the standard target and perturbation views. Timeline averages are a navigation aid;
target-level values remain available and are not replaced by averages.

## Comparing experiments

Completed runs can be reopened without inference. Select two to four saved runs to
compare method identity, model, input, N, seed, detector settings, target count, and
per-target distributions for persistence, confidence variation, class agreement,
entropy, and mean IoU.

For one image, the sequential batch control accepts values such as `5, 10, 20, 30`.
These runs execute one after another and reuse an unchanged loaded detector when safe.
They are never launched as parallel GPU jobs.

The saved data model includes method name, method identifier, method version, core and
dashboard schema versions, configuration, model, input identity, timestamp, and raw
results. A later real method such as Monte Carlo Dropout V2 can implement the method
interface and participate in comparisons without changing V1 results. V2 is not
implemented here.

## Saved outputs

Successful runs are published under:

```text
08_MODEL_UNCERTAINTY/dashboard/outputs/<unique-run-id>/
```

An image run contains:

```text
dashboard_metadata.json
summary.json
targets.csv
sample_metadata.json
diagnostics.json
samples/sample_*.jpg
previews/sample_*.jpg
```

A video run adds `video_summary.json`, `video_frames.csv`, and one complete result folder
for every selected frame. Comparisons export a CSV. Run identities use input, method,
UTC time, and a job ID; existing runs are never overwritten. Files are built inside an
owned `.staging` directory and the directory is atomically published only after success.

All runtime output is ignored by Git. Uploaded source files and model weights are not
copied into commits. The dashboard warns when an input path is inside the repository.

## Tests

The automated suite uses fake detectors, synthetic images, and small synthetic videos.
It does not require CUDA, a real model, ROS 2, a camera, or internet access.

```powershell
$Python = "$env:USERPROFILE\Desktop\UAV_YOLO_ENV\Scripts\python.exe"
$env:PYTHONPATH = "$PWD\08_MODEL_UNCERTAINTY\src;$PWD\08_MODEL_UNCERTAINTY\dashboard\src"
& $Python -m unittest discover -s .\08_MODEL_UNCERTAINTY\tests -v
& $Python -m unittest discover -s .\08_MODEL_UNCERTAINTY\dashboard\tests -v
```

## Current limitations

- V1 measures response stability only under its five configured mild perturbation
  families; it is not Monte Carlo Dropout.
- Greedy class-agnostic IoU matching can split, merge, or swap identities in crowded or
  overlapping scenes.
- Video frames are analyzed independently; the dashboard does not track identities
  through time.
- Detector and GPU kernels can introduce nondeterminism beyond the seeded input generator.
- Saved preview JPEGs are diagnostic artifacts and do not alter the source image or video.
- This is an offline R&D tool, not a safety certification or flight-control signal.
