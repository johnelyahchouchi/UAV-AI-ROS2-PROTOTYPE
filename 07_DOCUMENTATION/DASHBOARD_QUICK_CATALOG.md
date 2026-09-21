# Dashboard quick catalog

Use this guide beside the **UAV Perception and Mission Control** window. No UAV hardware is needed for MP4 testing.

## 1. Where to go

| Main tab / button | What it does |
| --- | --- |
| Dashboard | Checks Python, compute device, model trust, dependencies and running applications. Readiness is software readiness, not aircraft health. |
| Live Tester | Select your model and video, then configure inference and V1/V2 analysis. |
| Analysis Workspace | Watch live results, inspect V1/V2 samples and graphs, or extract a recorded video. |
| Secure Sender | Configure the existing video/detection connection to the ROS 2 receiver, including TLS certificates. Not needed for local MP4 analysis. |
| Mission Copilot | Run a simulated mission scenario and read explainable recommendations. It does not fly an aircraft. |
| Activity Log | Read application output and errors. Useful when a run cannot start. |
| Save settings | Remember configuration for the next launch; does not export analysis results. |
| Stop all | Stop managed applications and request cancellation of recorded extraction. |

## 2. Start with a video

Launch `01_WINDOWS_AI/launchers/Start_UAV_Prototype_GUI.bat`. In **Live Tester**, select a trusted base `.pt` checkpoint, choose **Select MP4**, enable continuous uncertainty, then launch. For V2, also select and enable its compatible trusted checkpoint. Missing V2 is shown as unavailable; V1 does not substitute for it.

| Setting | Read it this way |
| --- | --- |
| Base model / V2 model | Files containing learned model weights. V1 tests the base model; V2 uses the validated dropout model. |
| Registry / SHA-256 | Approved model fingerprints. Choosing a file alone does not approve it. See [model trust instructions](../00_PROJECT_GUIDE/MODEL_REGISTRY.md). |
| Source / loop | MP4 or screen capture; loop repeats video playback. Recorded extraction processes the selected MP4 once. |
| Confidence threshold | Minimum score for keeping a detection. Higher values remove more low-score results, but can also remove real objects. |
| NMS IoU | Overlap threshold used to suppress duplicate detections. Different from the matching IoU used by V1/V2. |
| Image size | Inference input size; the model may resize/pad the source. Exported box coordinates refer to source-image pixels. |
| Device | CPU, GPU or automatic selection. GPU availability depends on the installed environment. |
| FPS cap | Limits the live processing loop; it does not change the original video's recorded FPS. |
| Classes / tank-only | Filters the live display; the recorded panel uses its own detection-only pipeline and does not copy these display filters. |
| V1 samples / seed | Number of changed inputs, plus one clean baseline. Seed controls repeatable perturbation generation. |
| V2 passes | Number of stochastic evaluations of the same unchanged frame. More passes require more computation. |
| Matching IoU | Minimum spatial overlap used to associate boxes across samples/passes. Incorrect association can affect consistency scores. |

## 3. Live view and session controls

The main image shows current detections. The explanation panels replay **completed analysis samples** from a copied frame while playback continues. They can therefore refer to an older frame than the main video. Check the frame number, capture time and progress before comparing them.

- **V1:** clean image plus brightness, contrast, Gaussian blur and Gaussian noise variations. The parameters describe the actual input change. `gain=1` is neutral; brightness `offset` adds intensity; blur `kernel_size` is its pixel neighborhood; `sigma` controls blur/noise strength.
- **V2:** unchanged pixels evaluated repeatedly with internal dropout active. Differences come from the model's stochastic passes, not simulated blur. Cyan boxes show the selected pass; amber overlays show preceding pass boxes.
- **2x zoom:** enlarges a center crop for viewing. It does not change the input used for the measured result.
- **Pause / resume:** controls playback. **Save screenshot** saves a still image, not a video. **Stop session** ends the live run.
- **Dashboard preview up to 4 FPS:** the GUI picture refresh rate, separate from inference speed. **STALE** means the last preview is old; check pause state and Activity Log.
- **Start live session:** uses Live Tester settings. **Open saved session:** choose a folder containing `analysis.sqlite3`; no new inference runs. **Export full JSON:** saves all committed V1/V2 records available at export start, not video pixels.

## 4. Recorded extraction: every table column

One row means **one detection in one video frame**. Several rows can have the same frame number. A frame with no detections has no detection row.

| On screen (CSV name) | Meaning |
| --- | --- |
| frame (`frame_number`) | Frame position, starting at 1. |
| seconds (`timestamp_seconds`) | Position in the source video, beginning at 0 seconds; not time spent processing. |
| track (`track_id`) | Optional identity assigned by a tracker. Blank is expected here because this workspace uses detection-only mode. |
| class_id | Numeric label from that model's class list. It is not an object ID. |
| class (`class_name`) | Model-predicted category. It can be wrong. |
| confidence | Detection score from 0 to 1. `0.415361` is about 41.5% as a score, not a verified 41.5% probability of being correct. |
| x1, y1 | Top-left corner of the detection rectangle, in source pixels. |
| x2, y2 | Bottom-right corner of that rectangle, in source pixels. |

The image origin `(0,0)` is the **top-left**. **x increases to the right; y increases downward.** Decimal coordinates are normal model output.

```text
(0,0) --------------------------> x
  |        (x1,y1) +----------+
  |                |  object  |
  |                +----------+ (x2,y2)
  v y
```

Example from your screenshot: frame **23**, time **0.733333 s**, box **(486.43, 427.96, 597.73, 492.96)**. Its width is `x2-x1 = 111.30 px`; height is `y2-y1 = 65 px`; center is approximately `(542.08, 460.46)`. These are image positions, not GPS or real-world meters. The **2** is the class ID; the track column is blank.

**Choose MP4 → Start extraction** processes the video using the base model and the shared confidence, NMS IoU, image size, device and model-trust settings. V2 is not applied to this export. **Cancel** requests cleanup. Wait for **Complete** even if progress reaches 100% while encoding.

| Output / control | What you get |
| --- | --- |
| Open CSV | All detection rows; open in Excel or another analysis tool. |
| Open annotated video | Processed MP4 with boxes and labels drawn on it. |
| Output folder | Files from this completed extraction run. |
| Previous 100 / Next 100 | Pages of the displayed table, capped at 10,000 loaded rows. The full CSV is not capped by this display limit. |

## 5. Recorded summary below the table

| Field | Meaning |
| --- | --- |
| total_detections | Total rows across all frames, not unique objects. An object seen in 100 frames can contribute 100 detections. |
| counts_per_class | Detection rows for each predicted category, with the same repeated-frame counting. |
| average_confidence / maximum_confidence | Mean / highest score among detections; not model accuracy. `null` means no detections. |
| processed_frames | Number of frames processed, including frames with no detections. |
| processing_time_seconds | Elapsed time for the run, including final output preparation. Not video duration. |
| average_fps | Frames divided by core processing time; final video encoding is excluded. |
| source_fps / resolution | Original video frame rate and width × height in pixels. |
| inference_device / mode / model | Device used, detection/tracking mode, and checkpoint path. |
| detection_table_truncated | `true`: only the first 10,000 detections were loaded into the table; use the CSV for the rest. |

Your screenshot's **2,241 detections** across **2,196 processed frames** are observations, not a count of distinct vehicles or people.

## 6. V1 / V2 extraction and graphs

Select a sampled-frame row, then inspect **Sample inputs**, **Object metrics**, or **Complete JSON**. Sample/pass indices start at 0. V1 index 0 is the clean baseline. Each `target_id` associates boxes **within that analysis frame**; it is not a persistent video track.

| Metric / JSON names | How to read it |
| --- | --- |
| Persistence (`detection_persistence` / `persistence`) | Fraction of all samples/passes in which that object was detected. 8 detections in 10 passes = 0.8. Higher means more consistently present. |
| Class agreement (`class_agreement` / `winner_class_agreement`) | Fraction of detected samples whose label matches the most frequent label. Missing detections are excluded. |
| Dominant class / dominant winner class | Most frequent label for that object. Consistency does not prove the label is correct. |
| Mean reference IoU | Average box overlap with a reference rectangle: 0 = no overlap, 1 = identical. V1 uses the clean box if present, otherwise a mean box; V2 uses its mean representative box. This is not ground-truth overlap. |
| Minimum reference IoU (V2) | Worst overlap with that reference among detected passes. |
| Confidence mean / std | Average detection score / spread of scores. `std=0` means no measured variation, including when only one observation exists. V2 uses each pass's winning class score. |
| Box center / size std pixels | Variation in center position (`x`,`y`) or width/height (`x`,`y`), measured in pixels. Smaller means steadier boxes. |
| Class histogram / winner class distribution | Number of observed labels or pass-winning labels per class. |
| Class evidence share | V1: fraction of detected labels in each class. V2: average normalized confidence evidence across available per-pass class candidates. Neither is calibrated correctness probability. |
| Class / winner / evidence entropy bits | How spread out labels or evidence are. 0 means concentrated on one class; larger values indicate more disagreement. These are bits, not percentages. |
| Competition count / rate (V2) | Detected passes with competing classes / fraction of detected passes with this competition. |
| Existence / classification / localization status | Separate heuristic summaries of presence, label agreement and box stability. Read the underlying numbers and `interpretation` too. |
| sample_count, detected count, detected/missing indices | Total evaluations, evaluations with a matched object, and exactly which evaluations saw/missed it. |
| captured_utc | Computer's UTC time when the frame was decoded/captured; not the original recording date. |
| elapsed_seconds / duration_ms | Capture position in the current analysis session / time spent calculating this method's result. Session seconds differ from recorded-video seconds. |
| state / status | Processing state and result summary. Failed/unavailable is not evidence that there are no objects. |
| samples / detections / bbox | Every sample's transformation parameters and actual detections; `bbox` is `[x1,y1,x2,y2]` in source pixels. |

**Left graph: Detection and class consistency.** Cyan is mean persistence; amber is mean class agreement. A dip in persistence suggests objects disappear across evaluations. A dip in agreement suggests labels change when objects are detected.

**Right graph: Bounding-box consistency.** Cyan is mean reference IoU. A dip suggests box position/size varies more.

Both graphs use **session capture seconds horizontally** and a fixed **0–1 scale vertically**. Each point averages the objects in that sampled frame. The object mix may change between points; these are not individual-object trajectories. Gaps mean undefined/missing results, not zero. Only the latest 300 records per method appear; JSON retains the full history.

Example: persistence **0.5**, agreement **1.0**, IoU **0.9** means the object appears in half the evaluations, and when it appears its label and box are consistent. It does **not** mean perfect detection. V1 and V2 test different behavior and may use different models; do not interpret them as a direct accuracy contest.

## 7. Files and common messages

- Live journals and latest preview: `08_OUTPUTS/analysis_sessions/<session>/`. Keep the session folder to reopen it. The preview is a latest still, not a recorded live movie.
- Recorded CSV/video: `08_OUTPUTS/recorded_extraction/<run>/`.
- `N/A` / `null`: unavailable or undefined; not zero. `NO STOCHASTIC DETECTIONS`: no V2 detections in those passes, not proof of an empty scene.
- `STABLE`: outputs agree under this test; does not certify correctness. `VARIABLE`, `UNCERTAIN`, `UNSTABLE / REVIEW`: inspect the individual samples and metrics.
- Preview access error: the UI retries and keeps the last image. If it persists, check Activity Log and access to that session folder. Recorded extraction results are separate from the live preview.
- Battery health, sensor health, aircraft connection and aircraft online status are **not implemented in this dashboard yet**. Software readiness must not be read as hardware telemetry.

For launch details and limitations, see the [operator guide](UAV_PROTOTYPE_CONTROL_CENTER.md).
