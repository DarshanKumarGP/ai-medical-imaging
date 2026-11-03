# Model card

> **FOR EDUCATION ONLY. NOT FOR CLINICAL USE.**

This card covers the two models in this project. All numbers are measured on data the models were not trained on; where a sample is small, that is stated.

## 1. Chest X-ray classifier (`densenet121-rsna-v1`)

| Item | Detail |
|---|---|
| Task | Binary classification of frontal chest X-rays: lung opacity consistent with pneumonia, yes or no |
| Architecture | DenseNet121 pre-trained on ImageNet, final layer replaced by one output; 224x224 input; sigmoid score |
| Training | Two epochs with the backbone frozen, then full fine-tuning; class-weighted loss; mild augmentation (small rotation, shift, scale, brightness, contrast; no horizontal flip); early stopping on validation AUC (best epoch 4 of 7) |
| Explanation | Grad-CAM on the last DenseNet block |
| Decision rule | Flagged if score >= 0.4837 (threshold chosen on the validation split) |
| Output meaning | A **risk score**. Training used class weighting, so it is not a calibrated probability |

**Intended use:** demonstrating an AI-assisted review workflow (analysis, explanation, human approval, audit trail) for teaching.
**Out of scope:** any real diagnosis, triage or screening; other body parts or modalities; children (the data contains few young patients and no child-specific testing was done); decisions without a human reviewer.

### Data
RSNA Pneumonia Detection Challenge (stage 2 training set), 26,684 patients. Positive = a radiologist drew at least one "lung opacity" box. Split **by patient**: 18,678 train / 4,003 validation / 4,003 test, 22.5% positive in each.
Properties that matter: (1) the label is a **radiologist's opacity annotation**, not a confirmed pneumonia diagnosis; (2) according to the dataset description the negative class mixes normal films with abnormal films that have no opacity, so "negative" does not mean "healthy"; (3) annotations vary between readers, so labels are noisy; (4) one source, one labelling process, no external test set.

### Performance on the test split (4,003 patients)
AUC **0.887** (95% CI 0.876-0.899); sensitivity 0.871; specificity 0.721; precision 0.476 (expected at 22.5% prevalence); F1 0.616; TN 2235, FP 866, FN 116, TP 786.

### Subgroup results (one global threshold)

| Subgroup | n | AUC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|
| Female | 1681 | 0.888 (0.870-0.904) | 0.855 | 0.734 |
| Male | 2322 | 0.886 (0.870-0.899) | 0.883 | 0.711 |
| Age < 40 | 1309 | 0.900 (0.883-0.916) | 0.888 | 0.710 |
| Age 40-64 | 2151 | 0.887 (0.873-0.901) | 0.868 | 0.733 |
| Age 65+ | 543 | 0.850 (0.817-0.884) | 0.839 | 0.699 |
| View AP | 1833 | 0.831 (0.810-0.850) | 0.932 | 0.467 |
| View PA | 2170 | 0.872 (0.845-0.895) | 0.646 | 0.865 |

- **Sex:** no meaningful difference.
- **Age 65+:** lower AUC. The intervals only just touch and the group holds about 118 pneumonia cases, so this is suggestive, not conclusive.
- **View position (largest effect):** AP (portable, often sicker patients, 38.7% positive here) versus PA (standing, 8.8% positive). With one threshold the model flags about 69% of AP films with a specificity of 0.47, and misses about 35% of pneumonia on PA films. This is consistent with the model using the look of portable films as a cue for "sick" (not tested directly).
- **Tested mitigation:** one threshold per view (chosen on validation) cut PA misses to 14% and AP false alarms to 32%, but overall sensitivity fell from 0.871 to 0.814 with no gain in specificity. It was **not adopted**; details in `docs/TEST_RESULTS.md`.
- **Not auditable:** the data has no race, ethnicity, scanner or site information, so fairness across those groups is unknown.

### Explanation quality (Grad-CAM)
On 40 test pneumonia cases (35 flagged), the hottest point fell inside a radiologist's box in 54% of flagged cases (11% expected by chance). The heat map comes from a 7x7 feature map, so it is coarse and is **not an outline of disease**. It is drawn only for flagged scans, because the method rescales every map to 0-1 and would paint a red region on a normal-scored scan.

### Failure examples (from unseen scans)
- `68ceed79`: radiologists marked pneumonia, score 0.07, **missed**.
- `81d4ef3f`: marked normal, score 0.78, **false alarm** on a portable AP film; the heat lay at the lower left edge of the chest, outside the lung fields.
- Six unseen scans uploaded live: 4 correct, 1 missed, 1 false alarm, consistent with the measured rates.

## 2. CT spleen segmentation (`monai-unet3d-spleen-v1`)

| Item | Detail |
|---|---|
| Task | Segment the spleen in axial CT |
| Architecture | MONAI 3D U-Net (channels 16-256, residual units, batch norm), Dice loss, 96-voxel training patches, sliding-window inference, 150 epochs |
| Preprocessing | RAS orientation; resampled to 1.5 x 1.5 x 2.0 mm; window -57 to 164 HU scaled to 0-1; crop to foreground |
| Data | Medical Segmentation Decathlon Task09 Spleen: 41 volumes from one institution (per the dataset description); 29 train / 6 validation / 6 test |
| Test performance | Dice **0.925 ± 0.048** (per volume: 0.821, 0.926, 0.960, 0.943, 0.944, 0.954); best validation Dice 0.944 |
| Reported "confidence" | mean model probability inside the predicted mask |

**Intended and out-of-scope use:** same as above. One organ only; other organs or findings are not handled.

**End-to-end check:** through DICOM and Orthanc the pipeline gives the same grid and the same Dice as the original NIfTI files (0.8205 and 0.9599 versus 0.8207 and 0.9599 in the notebook).

**Failure example:** `spleen_26` (Dice 0.82): predicted volume 147 mL against 202 mL ground truth, 27% too small, while the reported confidence was **97%**. High confidence does not mean a correct outline. The viewer states this next to the number.

**Limits:** six test volumes (wide uncertainty on the Dice estimate); one institution and one scan type; no abnormal anatomy (for example very large spleens or surgery) was tested; resampling to 1.5 x 1.5 x 2 mm loses fine detail; volumes are measured on the resampled grid. A CT with fewer than 30 slices, uneven slice spacing or a non-axial orientation is skipped with a reason instead of being analysed.

## 3. System safeguards and their limits

- A "for education only, not clinical use" banner is always visible in the viewer.
- Every AI result needs a human decision. Decisions go to an append-only audit log (reviewer, decision, comment, AI score, model version, UTC time); the database refuses edits and deletes.
- **Reviewer names are typed in.** There is no login, so the log shows who *claims* to have decided. A real system needs authentication.
- The Orthanc demo login (`demo` / `demo`) is for local use only.
- The demo uses public research data. Patient IDs and names in the demo are random identifiers; no real patient data is stored.
- Models and data are versioned with DVC; library versions are pinned for the Docker images.

## 4. What would be needed before any real-world use

External validation on other hospitals and devices; calibration; per-site and per-view operating points; subgroup audits that include race, ethnicity and scanner; retraining with view-aware sampling; prospective evaluation with radiologists; authentication, privacy and security review; and regulatory approval. None of this has been done here.
