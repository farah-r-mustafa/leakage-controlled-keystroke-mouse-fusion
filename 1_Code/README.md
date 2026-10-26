# 1_Code — processing, training and evaluation pipeline

Code of the research platform used for the experiments (Phase C+, Python 3.12.10). Raw datasets and run outputs are not included.

| Folder | Stages | Content |
|---|---|---|
| `pipeline/02_KEYSTROKE_MOUSE/K01` | Stage 1 | Data validation of the UEBA dataset |
| `pipeline/02_KEYSTROKE_MOUSE/K02A` | Stage 2A | Session-level chronological split (manifests in `2_Splits/`) |
| `pipeline/02_KEYSTROKE_MOUSE/K02B`, `K02C` | Stages 2B–2C | Windowing, feature extraction, training-only preprocessing, tensors |
| `pipeline/02_KEYSTROKE_MOUSE/K03A` | Stage 3A | Construction of Balanced 1:1 and Impostor-All claims |
| `pipeline/02_KEYSTROKE_MOUSE/K03B`, `K03C` | Stages 3B–3C | Classical models (RF, SVM, …) and deep models (CNN, BiLSTM, CNN+BiLSTM) |
| `pipeline/02_KEYSTROKE_MOUSE/K04A`, `K04B` | Stages 4A–4B | Validation calibration and thresholds; single opening of the test part |
| `pipeline/02_KEYSTROKE_MOUSE/K51P`, `K51F` | Stage 5.1 | Statistical tests, bootstrap and deployment scorecard |
| `pipeline/01_BALABIT/B02`–`B05`, `B51` | Balabit | Preprocessing, windowing, baselines, deep models and statistics |

Each stage folder contains the Python script, a Windows `.bat` launcher and a `README_AR.md` (Arabic) with the run command. Local paths in the launchers and instructions were replaced by `<local_path>`; pass your own input and output paths.

Requirements: `requirements.txt` (core) and `requirements_research_deep.txt` (deep models). The experiments used PyTorch 2.7.0 (CPU-only build), scikit-learn 1.5.0 and NumPy 2.5.1.

Note: the scripts for the Aalto 136M experiments are not part of this package and are available from the corresponding author on request.
