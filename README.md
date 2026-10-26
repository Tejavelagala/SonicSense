# SonicSense

SonicSense is an environmental sound classification project using machine
learning and the UrbanSound8K dataset. The canonical application uses an
Improved Support Vector Machine (SVM) with a 29-feature audio pipeline.

## Target classes

The model predicts one of these ten UrbanSound8K classes:

1. `air_conditioner`
2. `car_horn`
3. `children_playing`
4. `dog_bark`
5. `drilling`
6. `engine_idling`
7. `gun_shot`
8. `jackhammer`
9. `siren`
10. `street_music`

## Canonical application

- [`app.py`](app.py) is the canonical Streamlit application.
- [`predict.py`](predict.py) is the canonical prediction module.
- The application loads:
  `models/svm_improved_official_fold10_model.joblib`
- The backup files [`app_improved.py`](app_improved.py) and
  [`predict_improved.py`](predict_improved.py) are retained for reference.

Launch the application from the project root:

```powershell
streamlit run app.py
```

The canonical app and model path have been verified in this project. The
command assumes that the required dependencies, model file, and local audio
resources are available.

## Environment setup

The existing project environment was verified with Python 3.12.10. A new
environment is not required for the existing project, but a separate
environment can be created for a new checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project has not been tested here in a newly created environment. The
commands above are the standard setup sequence for the documented files and
should be reviewed if the local Python installation uses a different command.

## Improved 29-feature pipeline

The Improved SVM uses these 29 features for each mono audio recording:

- 13 MFCC means: `mfcc_1_mean` through `mfcc_13_mean`
- 13 MFCC standard deviations: `mfcc_1_std` through `mfcc_13_std`
- `rms_mean`
- `zcr_mean`
- `spectral_centroid_mean`

Audio is loaded with the original sample rate preserved (`sr=None`) and
converted to mono. The predictor uses the same feature definitions and order
as the improved training pipeline.

## Project structure

```text
SonicSense/
├── app.py
├── predict.py
├── extract_improved_features.py
├── train_improved_svm_holdout.py
├── evaluate_improved_svm_all_folds.py
├── data/
│   ├── raw/UrbanSound8K/
│   │   ├── audio/fold1 ... fold10/
│   │   └── metadata/UrbanSound8K.csv
│   └── processed/
├── models/
│   └── svm_improved_official_fold10_model.joblib
└── reports/
```

The repository also contains original 16-feature SVM and Random Forest
baselines, diagnostic utilities, their training/evaluation scripts, and
historical reports. They are separate from the canonical Improved SVM path.

## Data requirements

The local dataset layout expected by the feature extraction and evaluation
scripts is:

```text
data/raw/UrbanSound8K/
├── audio/fold1/ ... audio/fold10/
└── metadata/UrbanSound8K.csv
```

The metadata and audio paths above have been verified in the existing project.
The raw audio dataset is not included in this documentation.

## Reproducing features and evaluations

The following scripts are the verified Improved SVM workflow:

1. Extract the improved features:

   ```powershell
   python extract_improved_features.py
   ```

   This reads UrbanSound8K and writes
   `data/processed/sonicsense_improved_features.csv`. It may also write
   `data/processed/improved_feature_extraction_errors.csv` if extraction
   errors occur.

2. Train and evaluate the official Fold 10 holdout:

   ```powershell
   python train_improved_svm_holdout.py
   ```

   This uses folds 1–9 for training and fold 10 for evaluation, then writes
   the model to
   `models/svm_improved_official_fold10_model.joblib` and the report to
   `reports/svm_improved_official_fold10_report.txt`.

   **Warning:** this script writes the canonical model path. Do not run it
   when the existing model must remain untouched unless the model artifact has
   been backed up or replacement is explicitly intended.

3. Run the ten-fold Improved SVM evaluation:

   ```powershell
   python evaluate_improved_svm_all_folds.py
   ```

   This writes `reports/svm_improved_all_folds_results.csv`. The script
   trains evaluation models for each fold and does not write the canonical
   model.

These commands have not been re-run as part of creating this documentation.
The results below are taken from the existing reports in this project.

## Verified results

### Official Fold 10 holdout

- Training recordings: 7895
- Test recordings: 837
- Features: 29
- Accuracy: **67.86%**
- Macro F1: **0.6859**

Fold 10 is the official known-class holdout used by the Improved SVM training
script and report.

### Ten-fold evaluation

The existing ten-fold results report approximately:

- Mean accuracy: **63.25%**
- Mean macro F1: **0.6354**

The exact per-fold values are available in
`reports/svm_improved_all_folds_results.csv`.

## Limitations

- Unknown-sound or out-of-distribution rejection is **not implemented**.
  Audio outside the ten target classes will still receive one of the known
  class labels.
- SVM decision scores are raw decision scores, not calibrated probabilities
  or confidence percentages.
- The three root-level sample recordings are exploratory examples, not a
  representative unknown-sound evaluation set.
- The reported random-split baseline results and official Fold 10 results use
  different evaluation protocols and should not be compared as equivalent
  measurements.

## UrbanSound8K attribution and licensing

This project uses UrbanSound8K version 1.0. The local dataset documentation
identifies Justin Salamon, Christopher Jacoby, and Juan Pablo Bello as the
dataset creators and states that the excerpts originate from recordings
uploaded to Freesound.

The included dataset documentation requests citation of:

> J. Salamon, C. Jacoby and J. P. Bello, “A Dataset and Taxonomy for Urban
> Sound Research,” 22nd ACM International Conference on Multimedia, Orlando,
> USA, Nov. 2014.

The local `UrbanSound8K_README.txt` states that UrbanSound8K is offered free
of charge for non-commercial use under the Creative Commons Attribution
Noncommercial License (BY-NC), version 3.0, and points to:

<http://creativecommons.org/licenses/by-nc/3.0/>

It also directs users to `FREESOUNDCREDITS.txt` for the attribution list.
Review the original dataset documentation and individual source requirements
before redistribution or commercial use. This project does not make any
additional licensing or permission claims.
