# SonicSense — GitHub Copilot Project Handoff

> Purpose: Give GitHub Copilot the project context, current state, constraints, and next steps for SonicSense.
> Local project path: `D:\Projects\SonicSense`
> Project deadline previously discussed: 28 October 2026. Confirm the current deadline before planning.
> Important: Treat the repository as the source of truth. Some details below come from previous work notes and must be verified against the actual files before making changes.

## 1. Role and working style

Act as a careful senior Python / machine-learning engineer and project mentor. Work directly in this repository, but make changes conservatively.

The user is tired of repeating setup and debugging steps. Avoid asking them to repeat checks or re-run completed setup work unless there is a clear reason. Inspect the current repository and terminal context first. Explain actions briefly and in plain language. Prefer small, reversible steps.

### Non-negotiable safety rules

1. **Do not delete, rename, overwrite, or replace working project files without explicit approval.**
2. Do not retrain, replace, or overwrite any existing model as a first step.
3. Do not change the existing prediction behavior or UI until a separate experiment has been tested and the user approves integration.
4. Do not recreate virtual environments, reinstall Python, or reinstall dependencies just because one terminal command fails. Diagnose the actual interpreter and environment first, and ask before making environment changes.
5. Do not commit the raw UrbanSound8K dataset, `.venv`, large generated artifacts, or model files to Git unless the user explicitly decides otherwise.
6. Do not assume the active terminal is using `D:\Projects\SonicSense\.venv`. In a previous session, PowerShell activated `D:\Projects\ChatFlow\.venv`, whose Python launcher referenced a missing Python 3.11 installation.
7. Before edits, inspect relevant files and Git status. If Git is available, create a branch or checkpoint only if appropriate and explain it; do not discard user changes.
8. For any proposed code change, show the files to be changed and why. Keep changes minimal. Run focused tests that do not alter existing data/models.
9. Never claim unknown-sound detection is accurate until it has been validated on held-out known examples and genuinely out-of-class unknown recordings.
10. If a dependency/environment problem blocks work, explain the blocker and propose the smallest safe diagnostic. Do not send the user through a long generic setup checklist.

## 2. Project goal

**SonicSense — Environmental Sound Classification Using Machine Learning**

A Python / librosa / scikit-learn / Streamlit project using UrbanSound8K to classify environmental audio into 10 predefined classes. The project has a working Streamlit application and an Improved SVM prediction pipeline. The user wants to make predictions more robust, especially for sounds outside the training classes.

The immediate goal is to design and evaluate **unknown-sound / out-of-distribution rejection** around the existing Improved SVM, while preserving the existing model and UI until the new detector is independently validated.

This is not the same as identifying every unknown sound by name. The first goal is to say that a recording may be outside the 10 known categories or the model is uncertain, instead of always presenting a misleading known-class label.

## 3. Dataset and known classes

Raw dataset location:
`data\raw\UrbanSound8K`

Expected structure includes metadata and audio folds 1–10. Metadata file:
`data\raw\UrbanSound8K\metadata\UrbanSound8K.csv`

Audio is organized under:
`data\raw\UrbanSound8K\audio\fold1` through `fold10`

UrbanSound8K's 10 class labels:

| classID | class |
|---:|---|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

Verify label strings and paths against the metadata/repository before using them in code.

The dataset does **not** include a separate unknown class. Official Fold 10 contains known classes and is intended for known-class evaluation, not as an unknown-sound test set.

## 4. Existing pipeline and files (verify names and implementation before edits)

### Original 16-feature pipeline
- `extract_all_features.py` → `data\processed\sonicsense_features.csv`
- `prepare_dataset.py` → `train_features.csv`, `test_features.csv`
- `train_svm.py` → `models\svm_model.joblib`
- `train_model.py` → `models\random_forest_model.joblib`
- `predict.py` and `app.py` use the original SVM.

### Improved 29-feature pipeline
- `extract_improved_features.py` → `data\processed\sonicsense_improved_features.csv`
- `train_improved_svm_holdout.py` → `models\svm_improved_official_fold10_model.joblib`
- `predict_improved.py` and `app_improved.py` use the Improved SVM.

The user reported that `app_improved.py` is running and predictions work. The UI was redesigned with a dark navy/cyan/violet gradient, upload/example mode, waveform visualization, audio information, and prediction display. Do not replace it without approval.

Existing model artifacts previously noted:
- `models\random_forest_model.joblib` (~62.66 MB), used by `compare_models.py`
- `models\svm_model.joblib` (~0.98 MB), used by `predict.py` and `compare_models.py`
- `models\svm_official_fold10_model.joblib` (~1.04 MB), generated by `evaluate_official_folds.py`; direct loading reference was not found in earlier inspection
- `models\svm_improved_official_fold10_model.joblib` (~1.36 MB), loaded by `predict_improved.py`

Keep both original and improved app/prediction files until the user explicitly approves cleanup.

Other notes:
- Root sample audio files previously observed:
  - `car-ignition-1.wav`
  - `gun-cocking-01.wav`
  - `mixkit-happy-child-laughing-2265.wav`
- `extract_features.py` appears to be a single-example/debug extractor, not a duplicate of `extract_all_features.py`.
- `inspect_audio.py` and `visualize_audio.py` appear to be standalone utilities.
- A stale `data\processed\feature_extraction_errors.csv` (~533 KB) contains repeated errors `'Pandas' object has no attribute 'class'`. User said renaming is not needed; leave it untouched.
- Raw dataset and `.venv` should stay local and be excluded from Git.

## 5. Improved feature definition

The improved pipeline uses 29 features:
- 13 MFCC means: `mfcc_1_mean` through `mfcc_13_mean`
- 13 MFCC standard deviations: `mfcc_1_std` through `mfcc_13_std`
- `rms_mean`
- `zcr_mean`
- `spectral_centroid_mean`

The current pasted implementations of `extract_improved_features.py` and `predict_improved.py` both use:
- `librosa.load(..., sr=None, mono=True)`
- MFCC with `n_mfcc=13`
- MFCC means and standard deviations
- mean RMS, mean zero-crossing rate, mean spectral centroid

They appear logically aligned in the snippets previously reviewed. Do not assert a feature-extraction mismatch without inspecting the current repository and proving it.

## 6. Current Improved SVM prediction function (previously reviewed)

The previous version of `predict_improved.py`:
- loads `models\svm_improved_official_fold10_model.joblib`
- extracts the 29 features above into a one-row pandas DataFrame with named columns
- predicts `predicted_class = model.predict(features)[0]`
- returns `(predicted_class, decision_scores)` where `decision_scores = model.decision_function(features)`

Important: decision scores are **not calibrated probabilities**. A large score should not be described as a probability or proof that a class is correct. If the model is a multiclass SVM, inspect its actual estimator/pipeline and scikit-learn configuration to understand the decision-score shape and semantics.

A prior UI concern: the UI expected a dictionary of scores, but `predict_sound()` returned raw NumPy decision scores. This may affect score display, not necessarily the predicted class. Verify current code before addressing it, and keep this issue separate from unknown-sound detection.

## 7. Evaluation results previously reported

These are results from earlier project work, not freshly verified in this session. Reproduce or confirm from repository artifacts before using them in formal reporting.

| Model / evaluation | Previously reported result |
|---|---:|
| Original SVM, random split | 90.56% accuracy |
| Random Forest, random split | 89.12% accuracy |
| Original SVM, official Fold 10 | 59.86% accuracy; macro F1 0.5704 |
| Improved SVM, official Fold 10 | 67.86% accuracy; macro F1 approximately 0.69 |
| Improved SVM, 10-fold mean | approximately 63.28% accuracy; mean macro F1 approximately 63.54% |

Do not compare random-split and official Fold 10 metrics as if they were the same evaluation. Clearly label each evaluation method. Check for leakage and confirm that preprocessing/scaling is fit only on training data.

## 8. Manual predictions observed in the current app

The user manually tested these root-level recordings in the running Streamlit app:

| Recording | Observed prediction | Interpretation |
|---|---|---|
| `car-ignition-1.wav` | `air_conditioner` | Likely a wrong known-class prediction; ignition is not a separate dataset class |
| `mixkit-happy-child-laughing-2265.wav` | `children_playing` | Reasonable/correct known-class prediction |
| `gun-cocking-01.wav` | `gun_shot` | Potentially misleading: a cocking sound is not the same event as a gunshot |

These are only three manually selected examples and are **not enough** to estimate overall accuracy or tune a rejection threshold. The two unexpected labels do not by themselves prove the inputs are out-of-distribution; the detector needs evaluation.

## 9. Unknown-sound detection: recommended engineering plan

### Objective
Wrap the current classifier with a separate reject/uncertainty layer. Keep the base SVM unchanged initially.

### Candidate signals
Evaluate, rather than blindly trust:
1. Multiclass SVM decision-score margins / relative scores.
2. Distance from the known training-feature distribution (e.g. standardized feature-space distance or class-conditional distance).
3. A separately trained out-of-distribution detector if sufficient representative unknown data becomes available.

Decision scores are not probabilities. A threshold must be selected empirically, not guessed.

### Data split discipline
- Use a validation set drawn only from appropriate training folds to choose/tune the rejection threshold.
- Preserve official Fold 10 as a final known-class test if that remains the agreed protocol.
- Acquire a separate unknown dataset that contains sounds outside the 10 UrbanSound8K classes (e.g. birdsong, rain, speech, ignition, gun cocking, household sounds, music), with licensing/source notes as appropriate.
- Do not label every audio outside UrbanSound8K as definitely unknown solely because it is from a different file source; verify its content and intended category.
- Include hard negatives and variation in recording quality/background noise.
- Report:
  - known-class accuracy before and after rejection
  - known false-rejection rate
  - unknown detection / rejection rate
  - confusion matrix and per-class metrics where applicable
  - performance by sound type, not only a single aggregate
  - number of recordings in each evaluation group

### Safe implementation sequence
1. Inspect repository and current Git status; don't edit yet.
2. Confirm the saved model object, class order, pipeline steps, feature schema, and training/evaluation split scripts.
3. Design an isolated experiment. If a separate test script is useful, ask/confirm before adding it because the user previously said they prefer manual verification and specifically rejected creating a testing script at that moment.
4. Start with a written experiment plan or notebook only after approval; do not modify `predict_improved.py`, `app_improved.py`, or any `.joblib` file initially.
5. Use manual test results and a sufficiently sized, labeled dataset to evaluate candidate rejection rules. Do not tune to just the three example clips.
6. Report trade-offs: increasing unknown rejection can also reject valid known sounds.
7. Only after the user approves and results look reasonable, integrate the reject layer behind a small, reversible change. Preserve old behavior behind a toggle or separate function if practical.
8. Test the UI with known and unknown audio and verify the score UI. Avoid claiming that unknown sounds are identified by name; rejection only indicates the sound may not fit the known classes.

## 10. Environment context

A previous PowerShell session showed:
- Prompt initially activated `D:\Projects\ChatFlow\.venv`, not `D:\Projects\SonicSense\.venv`.
- Running `python` gave: `No Python at '"C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe'`.
- No fix was completed. Do not assume the current terminal or environment is the same.
- The user is frustrated by repeated environment troubleshooting. Avoid restarting the setup process. If a command is necessary, first inspect current workspace/terminal and explain why the command is needed. Do not delete/recreate `.venv` or install Python without explicit permission.

## 11. Other priorities before final submission

Subject to the actual deadline and user's priorities:
1. Preserve current working app/model.
2. Validate current model and UI end-to-end.
3. Decide whether unknown rejection is feasible with available data and time.
4. Generate verified evaluation outputs: confusion matrix, per-class precision/recall/F1, known vs unknown rejection metrics.
5. Add clear app messaging for uncertain/out-of-scope audio if validated.
6. Finalize README/setup steps, screenshots, methodology, limitations, results, and presentation/report as required.
7. Confirm `.gitignore` excludes `.venv`, raw data, caches, and unintended large generated files.

## 12. How Copilot should begin

Start by responding with a short repository audit plan, then inspect the actual files and Git status without modifying anything. Confirm:
- Current branch and working tree changes.
- Which app is used and which model it loads.
- Feature names/order at training and inference.
- Model class labels and decision-function behavior.
- Which test/evaluation scripts exist and how splits are formed.
- Current `.gitignore` protections.

Then present a concise status report: verified facts, discrepancies, risks, and a recommended next step. **Do not start editing, run training, create new scripts, change the environment, or install packages until the user approves the plan.**

The user's explicit preference is: careful step-by-step work, no repeated setup steps, preserve existing files, and no changes to the working app/model without testing and approval.
