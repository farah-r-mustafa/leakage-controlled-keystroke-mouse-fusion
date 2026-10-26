# Leakage-Controlled Comparison of Unimodal and Fused Keystroke–Mouse Biometrics

Supplementary material for the manuscript:

> F. R. Mustafa and B. I. Farhan, "A Leakage-Controlled Comparison of Unimodal and Fused Keystroke–Mouse Biometrics for Continuous Authentication in Zero Trust," *Journal of Engineering* (University of Baghdad), manuscript ID 4629, under review.

If you use any part of this material, please cite the article above.

## Contents

| Folder | Content |
|---|---|
| `1_Code/` | Processing, windowing, feature-extraction, training and evaluation code (`research_core/`), with `requirements.txt` and `requirements_research_deep.txt`. Raw data are not included. |
| `2_Splits/` | Frozen train / validation / test split manifests (session-level split before windowing). |
| `3_Results_and_Statistics/` | Per-user and user-macro test metrics, statistical tests (sign-flip with Holm correction, McNemar, bootstrap), deployment scorecard, integrity checks and SHA-256 file manifests. |
| `4_Matched_Window_Ablation/` | Script and results of the matched-window ablation (Random Forest retrained on identical windows with keyboard-only, mouse-only and fused inputs). |

## Datasets (not redistributed)

| Dataset | Source | Use in this study |
|---|---|---|
| Balabit Mouse Dynamics Challenge | Public dataset (Antal and Egyed-Zsigmond, 2018) | Mouse only, 10 users, labelled training sessions |
| Aalto 136M Keystrokes | Public dataset (Dhakal et al., 2018) | Keyboard only, 100 participants sampled with seed 1362026 |
| Keystroke and Mouse Dynamics for UEBA | Public dataset (Martín et al., 2021) | Paired keyboard and mouse, 11 users |

Download the datasets from their original sources and place them as described in `1_Code/README_AR.md`.

## Software and hardware

Windows 11, Python 3.12.10, PyTorch 2.7.0 (CPU-only build), scikit-learn 1.5.0, NumPy 2.5.1.
Intel Core i7 CPU, 8 GB RAM, no GPU.

## Notes

- The protocol named "Enrolled Impostor-All" in the article is labelled `realistic_all_impostor` in the code and output files.
- All thresholds and calibrators were fitted on the validation part and frozen before the test part was opened.

## License

Released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0). See `LICENSE`.

## Contact

Corresponding author: Baraa I. Farhan, Department of Computer Science, College of Education for Pure Sciences, University of Wasit, Kut, Iraq.
