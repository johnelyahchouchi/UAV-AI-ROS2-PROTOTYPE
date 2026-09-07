# MC Dropout V2 research artifact

`montecarlo_v2.ipynb` is the original Colab training, architecture-audit, and
validation record imported from the downloaded `montecarlov2.ipynb`. Its retained
SHA-256 is
`3C3CF7B02FA2D5F1FE112A3696B4244DED54D38FC311B54A2871A1FB4F02D08D`.
The separately downloaded `montecarlov2 (1).ipynb` was also inspected: its cells,
outputs, and metadata were identical; only JSON serialization whitespace differed.
No credentials or secrets were found in the retained notebook.

The notebook is research provenance, not an application dependency. Colab and Google
Drive paths inside it intentionally record the experiment environment. Portable runtime
code must not import or execute notebook cells.

## Recovered validated method

- Ultralytics `8.4.107` with six `Dropout2d(p=0.20)` modules.
- One dropout immediately before the final prediction `Conv2d` in each of the three
  `cv2` regression and three `cv3` classification scales.
- Head-only fine-tuning for three epochs with layers 0–21 frozen, AdamW,
  `lr0=0.0001`, `lrf=0.1`, weight decay `0.0005`, one warm-up epoch, cosine learning
  rate, and mosaic disabled.
- At inference the detector is globally in evaluation mode, all BatchNorm modules
  remain in evaluation mode, and only the six `Dropout2d` modules are active.
- One preprocessed tensor is held fixed for 20 lower-level model forwards. The
  high-level prediction method is not used because it can reset dropout state.
- Post-processing uses confidence `0.25`, NMS IoU `0.45`, and class-agnostic spatial
  object association. Competing class hypotheses are retained per pass.

The notebook recorded these historical validation results:

| Experiment | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| V1 baseline | 0.682 | 0.531 | 0.567 | 0.374 |
| Final V2 | 0.678 | 0.523 | 0.574 | 0.375 |

V2 approximately preserved baseline detector performance while enabling stochastic
inference. The small metric differences do not establish that V2 is more accurate.

## Artifact lineage

```text
montecarlo_v2.ipynb
  original Colab research/training/validation record
        |
        +--> ../src/uav_uncertainty/mc_dropout_v2.py
        |      detector-independent validated runtime extraction
        |
        +--> ../src/uav_uncertainty/mc_dropout_ultralytics.py
               trusted Ultralytics lower-level forward adapter

external military_kaggle_v2_mcdo.pt
  trained checkpoint; intentionally excluded from source Git
```

The validated checkpoint was produced at:

`/content/drive/MyDrive/UAV_MC_DROPOUT_V2/headonly_p020_lr1e4_e3/weights/best.pt`

It was not available on this workstation during integration. Download that exact
artifact to external model storage, independently verify its provenance and SHA-256,
add the real digest and size to the trusted model registry, then set
`UAV_MCDO_V2_MODEL_PATH` to its absolute path. Never substitute or rename V1 weights
as V2.
