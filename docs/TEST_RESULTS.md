# Test results

Everything below was produced by running the project. Dates: 29 Oct to 3 Nov 2026.

## 1. Model performance (held-out data only)

### Chest X-ray classifier (RSNA Pneumonia Detection)
DenseNet121, ImageNet-pretrained. Two epochs with the backbone frozen, then fine-tuning with early stopping (stopped after epoch 7, best validation AUC 0.876 at epoch 4). Class-weighted loss, mixed precision.

| Item | Value |
|---|---|
| Split (by patient) | train 18,678 / validation 4,003 / test 4,003, pneumonia rate 0.225 in each |
| Decision threshold (chosen on validation) | 0.4837 |
| **Test AUC** | **0.887** |
| Accuracy | 0.755 |
| Sensitivity (recall) | 0.871 (786 of 902 pneumonia cases found) |
| Specificity | 0.721 |
| Precision | 0.476 (expected with 22.5% prevalence) |
| F1 | 0.616 |
| Confusion matrix (test) | TN 2235, FP 866, FN 116, TP 786 |

Plots: `docs/training_results.png` (loss, validation AUC, ROC, confusion matrix). Raw numbers: `docs/metrics.json`, `docs/history.json`.

### Grad-CAM localisation check
40 pneumonia patients from the **test split**; 35 were flagged. For the flagged cases, the hottest point of the heatmap fell inside a radiologist's box in **54%** (chance level: 11%, the share of the image covered by boxes). 27% of the total heat lay inside the boxes. Small sample: quote it as "about half, on 40 cases". Raw rows: `docs/gradcam_eval.csv`, `docs/gradcam_eval.json`.

### CT spleen segmentation (Medical Segmentation Decathlon Task09)
MONAI 3D U-Net (channels 16-256, residual units), Dice loss, 96-voxel patches, sliding-window inference, 150 epochs.

| Item | Value |
|---|---|
| Volumes | 41 total: 29 train / 6 validation / 6 test (fixed seed) |
| Best validation Dice | 0.944 |
| **Test Dice (mean ± sd, 6 volumes)** | **0.925 ± 0.048** |
| Per test volume | 0.821, 0.926, 0.960, 0.943, 0.944, 0.954 |

Plots: `docs/segmentation_curves.png`, `docs/segmentation_examples.png`.

## 2. System-level validation

| Check | Result |
|---|---|
| 60 X-rays sent through Orthanc, analysed by the service | 60 of 60 done, 0 errors; scores reproduce direct model runs |
| Test-split patients among them | sensitivity 0.88 (36 of 41) |
| Heatmap files | 45 stored = number flagged; 0 missing files |
| **CT through DICOM and Orthanc vs original NIfTI** (`src/check_ct.py`, `docs/ct_dicom_validation.txt`) | grid shapes identical; maximum image difference 0.0020 (8-bit rounding); Dice 0.8205 and 0.9599 versus 0.8207 and 0.9599 in the training notebook |
| Volume check | spleen_29: AI 187 mL vs ground truth 183 mL. spleen_26: AI 147 mL vs 202 mL (27% too small), while the model reported 97% confidence |
| Short CT (1 slice) | skipped with reason "needs at least 30 slices"; Approve/Reject disabled |
| 6 new unseen X-rays uploaded while the service was running | appeared in the viewer without a page refresh (60 to 66 analysed); 4 of 6 correct (1 missed pneumonia, 1 false alarm), consistent with the measured sensitivity and specificity |

## 3. Automated tests (`pytest -q`: **23 passed**)

| File | Tests | What they cover |
|---|---|---|
| `tests/test_api.py` | 10 | health and disclaimer; sorting and filtering; detail and 404; image serving; approve writes an audit entry (score, model version, time); re-review keeps full history; invalid input rejected (422); skipped studies cannot be reviewed (409); audit table refuses UPDATE and DELETE; no delete/edit endpoints (405) |
| `tests/test_ct_core.py` | 8 | DICOM series to volume round trip (orientation, origin, spacing); slice order independence; rescale slope and intercept; skip rules (single slice, uneven spacing, duplicate positions, non-axial); slice rendering orientation and overlay alignment |
| `tests/test_ct_api.py` | 5 | CT detail fields; slice scan and mask PNGs; out-of-range and missing-volume 404s; CT review goes to the audit log; skipped CT explains why and cannot be reviewed |

Frontend: `npm run lint` clean, `npm run build` succeeds (also executed inside the Docker build).

## 4. Manual test log (viewer)

| Test | Result |
|---|---|
| Click quickly through 20+ studies for a minute | every scan and heatmap loaded |
| Approve, then Reject the same study | both entries kept in the history, list status updated, counters updated |
| Missing reviewer name | red message, nothing saved |
| Filters (All / Flagged / Pending), prefix search, "Next pending study" | correct |
| API stopped | red banner, recovers by itself when the API returns |
| Upload a file that is already stored | no duplicate study |
| Live upload of 6 new X-rays | processed and shown automatically |
| CT: slice slider, jump button, overlay toggle and opacity | working |

## 5. Docker

| Check | Result |
|---|---|
| `docker compose up -d --build` | four containers: orthanc, inference, api, frontend |
| Fresh database | all 69 studies in Orthanc processed (66 X-rays, 2 CTs segmented, 1 short CT skipped); 0 approved / 0 rejected |
| Scores in containers vs laptop | identical for X-rays; CT 147 mL and 187 mL, identical |
| Persistence | a decision (Dr. Kiran, spleen_29) survived `docker compose down` and `up -d` |
| Orthanc stopped for 25 s, then restarted | the inference service logged "Orthanc not reachable ... retrying" instead of crashing, and resumed by itself after the restart (no further errors in the following 30 s) |
| Resources at idle | inference 433 MiB, api 47 MiB, orthanc 17 MiB, frontend 11 MiB. Both CT volumes completed within the 2.8 GB limit of the inference container. |

## 6. Subgroup (bias) analysis, X-ray classifier, test split (4,003 patients)

Sex, age and view position were read from the DICOM headers. One global threshold (0.4837). 95% confidence intervals by bootstrap (300 resamples). Table: `docs/subgroup_metrics.csv`.

| Subgroup | n | Pneumonia rate | AUC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| All | 4003 | 0.225 | 0.887 (0.876-0.899) | 0.871 | 0.721 |
| Female | 1681 | 0.217 | 0.888 (0.870-0.904) | 0.855 | 0.734 |
| Male | 2322 | 0.231 | 0.886 (0.870-0.899) | 0.883 | 0.711 |
| Age < 40 | 1309 | 0.245 | 0.900 (0.883-0.916) | 0.888 | 0.710 |
| Age 40-64 | 2151 | 0.215 | 0.887 (0.873-0.901) | 0.868 | 0.733 |
| Age 65+ | 543 | 0.217 | 0.850 (0.817-0.884) | 0.839 | 0.699 |
| View AP (portable) | 1833 | 0.387 | 0.831 (0.810-0.850) | 0.932 | 0.467 |
| View PA | 2170 | 0.088 | 0.872 (0.845-0.895) | 0.646 | 0.865 |

Findings: no meaningful difference by sex; lower AUC for age 65+ (intervals only just touch, about 118 pneumonia cases); a large difference by **view position**: the single threshold over-flags AP films (69% flagged, specificity 0.47) and under-detects pneumonia on PA films (sensitivity 0.65).

### Experiment: one threshold per view position
Thresholds chosen on the **validation** split (best sensitivity + specificity balance): AP 0.7517, PA 0.2488. Evaluated on the untouched test split (`docs/view_threshold_experiment.csv`).

| Setting | n | Sensitivity | Specificity | Precision | Share flagged |
|---|---|---|---|---|---|
| AP, one threshold (0.4837) | 1833 | 0.932 | 0.467 | 0.525 | 0.687 |
| AP, own threshold (0.752) | 1833 | 0.801 | 0.683 | 0.615 | 0.505 |
| PA, one threshold (0.4837) | 2170 | 0.646 | 0.865 | 0.317 | 0.180 |
| PA, own threshold (0.249) | 2170 | 0.859 | 0.748 | 0.249 | 0.306 |
| All, one threshold | 4003 | 0.871 | 0.721 | 0.476 | 0.412 |
| All, view-specific thresholds | 4003 | 0.814 | 0.725 | 0.462 | 0.397 |

Result: the errors are redistributed more evenly across views (PA misses fall from 35% to 14%, AP false alarms fall from 53% to 32%), but overall sensitivity drops from 0.871 to 0.814 with no gain in specificity. **Not adopted** in the deployed system: the single threshold keeps the higher overall sensitivity and is simpler. The result is documented as a tested mitigation and as evidence that retraining with view-aware sampling and external validation are the right next steps.

## 7. Known failures and limits

- One missed pneumonia and one false alarm among the six unseen X-rays (see above).
- CT volume can be wrong while confidence is high (spleen_26).
- Heatmaps are coarse (7x7 feature map) and are not outlines of disease.
- Trained and tested on one dataset per task; no external validation.
- Details and subgroup analysis: `MODEL_CARD.md`.
