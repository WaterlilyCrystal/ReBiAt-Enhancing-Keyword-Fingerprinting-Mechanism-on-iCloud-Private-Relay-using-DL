# ReBiAt: Residual BiGRU Attention for Keyword Traffic Fingerprinting

ReBiAt is a deep-learning system for keyword traffic fingerprinting on iCloud Private Relay (APR) traffic. It operates in three stages: 
(1) preprocessing raw PCAP captures into fixed-length packet sequences (L=500) and 15-dimensional global feature vectors; (2) extracting representations with a ResNet-10 frontend for local burst features and a BiGRU with temporal attention for longer-range structure; (3) classifying keywords with a softmax head in closed-world mode and rejecting out-of-distribution traffic in open-world mode using calibrated scores (max-softmax, energy, Mahalanobis). The system is evaluated on a macOS dataset of 50 monitored keywords with ~500 traces each (~25,000 labeled traces total), and is compared against Var-CNN and NetCLR across four scenarios: closed-world accuracy, open-world recognition, concept drift adaptation, and defense robustness (FRONT, WTF-PAD, BurstGuard).

---
## Requirements

Python 3.10+, PyTorch 2.x, scikit-learn, numpy, scipy, scapy, tqdm, joblib, matplotlib.

```bash
pip install torch torchvision scikit-learn numpy scipy scapy tqdm joblib matplotlib
```

---

## Repository Structure

| File | Role |
|------|------|
| `extract_features_v2.py` | Feature extraction from raw PCAP files, outputs .npz dataset files. |
| `augmentation.py` | On-the-fly data augmentation (timing jitter, size jitter, packet drop, mixup). |
| `feature_selection.py` | Offline global-feature importance study using ANOVA, mutual information, and RandomForest. |
| `train_resnet_bigru.py` | Train the proposed ReBiAt model (ResNet-10 + BiGRU + Attention). |
| `baselines.py` | PyTorch implementations of Var-CNN and NetCLR baselines plus a unified model dispatcher. |
| `train_baselines.py` | Train Var-CNN or NetCLR with the same protocol as the thesis model. |
| `open_world.py` | OOD scoring functions (max-softmax, energy, Mahalanobis) and evaluation metrics. |
| `open_world_pipeline.py` | End-to-end open-world evaluation pipeline. |
| `drift_pipeline.py` | Concept drift detection and temporal adaptation experiment. |
| `defenses.py` | Post-hoc traffic-analysis defense simulators (FRONT, WTF-PAD, BurstGuard). |
| `eval_defended.py` | Non-adaptive attacker evaluation on defended traffic. |
| `collect_defense_results.py` | Aggregate defense overhead and accuracy results into a summary table. |
| `compare_all_methods.py` | Tabulate all three methods across all four evaluation scenarios. |
| `visualize.py` | Generate all thesis figures (closed-world metrics, per-class tables, open-world score histograms, drift charts, defense bars). |

---

## Quick Start

Replace the placeholder paths with your actual directories before running.

```bash
# 1. Extract features from raw PCAPs
#    --root: folder containing one subfolder per keyword, each with .pcap files
#    --out:  output .npz file path
python extract_features_v2.py \
    --root ./data/keyword-pcaps \
    --out  ./features/dataset.npz

# 2. Train the ReBiAt model
#    --npz:         path to the .npz produced in step 1
#    --results_dir: directory where checkpoints and result JSON are saved
python train_resnet_bigru.py \
    --npz         ./features/dataset.npz \
    --results_dir ./results

# 3. Closed-world evaluation runs automatically at the end of training.
#    Results are saved to ./results/run_results.json

# 4a. Open-world evaluation from an already-extracted unknown .npz
#     --known_npz:   .npz of labeled (monitored) traffic
#     --unknown_npz: .npz of unlabeled unseen-keyword traffic
#     --checkpoint:  best_model.pt from step 2
#     --results_dir: output directory
python open_world_pipeline.py \
    --known_npz       ./features/dataset.npz \
    --unknown_npz     ./features/openworld_unknown.npz \
    --checkpoint      ./results/best_model.pt \
    --results_dir     ./results

# 4b. If the unknown traffic is still raw PCAP, open_world_pipeline.py can
#     extract it once and cache the resulting unknown .npz:
python open_world_pipeline.py \
    --known_npz        ./features/dataset.npz \
    --unknown_pcap_dir ./data/unknown-pcaps \
    --unknown_npz      ./features/openworld_unknown.npz \
    --checkpoint       ./results/best_model.pt \
    --results_dir      ./results

# 5. Concept drift experiment (two capture sessions required).
#    To reproduce the thesis few-shot setting, use --n_shots 20.
python drift_pipeline.py \
    --s1_npz      ./features/session1_10kw.npz \
    --s2_npz      ./features/session2_10kw.npz \
    --s1_checkpoint ./drift_results/session1_10kw_resnet_bigru.pt \
    --results     ./drift_results \
    --n_shots     20 \
    --strategies  F1,F3,F3_AUG,F_TEMP,F_TPROTO

# 6. Defense robustness evaluation
#    Run defenses.py first (produces defended .npz files), then evaluate:
python eval_defended.py \
    --checkpoint    ./results/best_model.pt \
    --defended_npz  ./results/defended_front.npz \
    --loader_workers 0   # set 0 on Windows to avoid multiprocessing paging errors

#    Aggregate overhead + accuracy into a summary table:
python collect_defense_results.py \
    --clean_results ./results/run_results.json \
    --defense front:./results/defended_front.npz:./results_front:./results/nonadaptive_front.json \
    --out ./results/defense_summary

# 7. Compare all methods across scenarios
#    Pass per-method result files for each scenario:
python compare_all_methods.py \
    --closed    ReBiAt:./results/run_results.json \
    --openworld ReBiAt:./results/open_world_results.json \
    --drift     ReBiAt:./drift_results/drift_report.json \
    --defense   ReBiAt:./results/defense_summary.json \
    --out_dir   ./results_comparison
```

---

## Running on Kaggle

The `_kaggle.py` source files in the parent directory (`../`) were the original Kaggle notebook cells adapted for the Kaggle environment (different paths, GPU session constraints, inline outputs). For the full Kaggle workflow, including dataset mounting, output paths, and notebook execution order, refer to `../BASELINE_COMPARISON_RUNBOOK.md`.

---

## Data

The expected dataset layout is one folder per keyword, each containing raw `.pcap` or `.pcapng` capture files. Class names are inferred automatically from folder names using `os.listdir()`.

```
pcap_dataset/
    apple music/
        trace_001.pcap
        trace_002.pcapng
        ...
    facebook/
        trace_001.pcap
        ...
    ...
```

`extract_features_v2.py` reads this layout and writes one dataset `.npz` containing `X_seq`, `X_global`, `y`, `classes`, and per-sample provenance metadata (`file_paths`, `capture_start_time`, `capture_end_time`). Training scripts then create train/val/test splits. For temporal evaluation, rebuild datasets with this version of the extractor and run training with `--chrono_split`; leave `--shuffle` off so the saved metadata remains directly auditable.

The runtime model does not perform per-inference feature selection. The 15 global features are curated offline once via `feature_selection.py`, then standardized on the training split and reused unchanged for validation, test, open-world, defense, and drift evaluation.

---

## Citation / References

### This work

Nguyen Phuong Linh. "Keyword Traffic Fingerprinting on Apple iCloud Private Relay Using Deep Learning." Bachelor Thesis, Bach Khoa Cyber Security Center (BKCS), 2025.

---

### Open-source software

The table below lists every third-party library used in this repository with its licence. Scapy is GPL v2; all other dependencies are permissive.

| Library | Licence | Used in |
|---------|---------|---------|
| PyTorch ≥ 2.0 | BSD-3-Clause | model training and inference (`train_resnet_bigru.py`, `train_baselines.py`, `open_world.py`, `drift_pipeline.py`, `eval_defended.py`) |
| NumPy | BSD-3-Clause | array operations (all modules) |
| SciPy | BSD-3-Clause | covariance estimation, IAT statistics (`open_world.py`, `extract_features_v2.py`) |
| scikit-learn | BSD-3-Clause | metrics, StandardScaler, ANOVA/MI/RF feature selection (`feature_selection.py`, `train_resnet_bigru.py`, `drift_pipeline.py`) |
| Scapy ≥ 2.5 | **GPL v2** | PCAP parsing (`extract_features_v2.py`) |
| Optuna | MIT | Bayesian hyperparameter search (`train_resnet_bigru.py`) |
| Matplotlib | PSF / BSD-compatible | all figures (`visualize.py`) |
| tqdm | MIT / MPL 2.0 | progress bars (`defenses.py`, `extract_features_v2.py`) |
| joblib | BSD-3-Clause | parallel PCAP extraction (`extract_features_v2.py`) |

**Note on Scapy (GPL v2):** redistribution of this repository in binary form, or linking it into a proprietary product, requires compliance with the GNU General Public License v2. For academic and research use (running scripts locally) no additional obligations apply beyond attribution.

Library paper citations:

- Adam Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." _NeurIPS_, 2019.
- Charles R. Harris et al. "Array programming with NumPy." _Nature_ 585, 2020.
- Pauli Virtanen et al. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." _Nature Methods_ 17, 2020.
- Fabian Pedregosa et al. "Scikit-learn: Machine Learning in Python." _JMLR_ 12, 2011.
- Philippe Biondi and the Scapy community. _Scapy_ (packet manipulation library). https://scapy.net.
- Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. "Optuna: A Next-generation Hyperparameter Optimization Framework." _KDD_, 2019.
- John D. Hunter. "Matplotlib: A 2D Graphics Environment." _Computing in Science & Engineering_ 9(3), 2007.

---

### Model architecture

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." _CVPR_, 2016.  
  ResNet-10 backbone with residual connections (`train_resnet_bigru.py`).

- Jie Hu, Li Shen, and Gang Sun. "Squeeze-and-Excitation Networks." _CVPR_, 2018.  
  SE channel-recalibration blocks inside each ResNet residual block.

- Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." _EMNLP_, 2014.  
  Gated Recurrent Unit (GRU) used as the bidirectional temporal encoder.

- Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate." _ICLR_, 2015.  
  Soft temporal-attention pooling over BiGRU hidden states.

---

### Training

- Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. "Focal Loss for Dense Object Detection." _ICCV_, 2017.  
  Focal loss with γ=2 used as the default training criterion (`FocalLoss` in `train_resnet_bigru.py`).

- Rafael Müller, Simon Kornblith, and Geoffrey Hinton. "When Does Label Smoothing Help?" _NeurIPS_, 2019.  
  Label smoothing applied inside both `FocalLoss` and `CrossEntropyLoss`.

- Ilya Loshchilov and Frank Hutter. "Decoupled Weight Decay Regularization." _ICLR_, 2019.  
  AdamW optimizer used throughout training.

- Ilya Loshchilov and Frank Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." _ICLR_, 2017.  
  Cosine-annealing learning-rate schedule (`CosineAnnealingLR`).

- Paulius Micikevicius et al. "Mixed Precision Training." _ICLR_, 2018.  
  `torch.amp.autocast` + `GradScaler` used for GPU training.

- Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. "mixup: Beyond Empirical Risk Minimization." _ICLR_, 2018.  
  Mixup augmentation in `augmentation.py` and `train_resnet_bigru.py`.

---

### Global feature set inspirations

- Andriy Panchenko, Lukas Niessen, Andreas Zinnen, and Thomas Engel. "Website Fingerprinting in Onion Routing Based Anonymization Networks." _WPES_, 2011.  
  Packet-size histogram bins in the global feature vector.

- Mohammad Saidur Rahman, Prerana Kundnani, Mohsen Imani, and Matthew Wright. "Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks." _PETS_, 2020.  
  IAT mean/std and response-latency features in the global vector.

- Tao Wang, Xiang Cai, Rishab Nithyanand, Rob Johnson, and Ian Goldberg. "Effective Attacks and Provable Defenses for Website Fingerprinting." _USENIX Security_, 2014.  
  Directional byte-ratio and burst-count features in the global vector.

---

### Open-world / OOD detection

- Dan Hendrycks and Kevin Gimpel. "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." _ICLR_, 2017.  
  Maximum softmax probability (MSP) score (`score_softmax` in `open_world.py`).

- Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. "Energy-based Out-of-distribution Detection." _NeurIPS_, 2020.  
  Energy score (`score_energy` in `open_world.py`).

- Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." _NeurIPS_, 2018.  
  Mahalanobis distance with pooled class covariance (`fit_mahalanobis`, `score_mahalanobis_loader` in `open_world.py`).

---

### Concept-drift adaptation

- Jake Snell, Kevin Swersky, and Richard Zemel. "Prototypical Networks for Few-Shot Learning." _NeurIPS_, 2017.  
  Nearest-class-mean cosine classifier used in the `F_TPROTO` adaptation strategy (`compute_prototypes`, `evaluate_prototype` in `drift_pipeline.py`).

---

### Baseline models

- Sanjit Bhat, David Lu, Albert Kwon, and Srinivas Devadas. "Var-CNN: A Data-Efficient Website Fingerprinting Attack Based on Deep Learning." _PETS_, 2019.  
  Var-CNN architecture implemented in `baselines.py`; trained via `train_baselines.py`.

- Alireza Bahramali, Ramin Khalili, Amir Houmansadr, Dennis Goeckel, and Don Towsley. "Robust Adversarial Attacks Against DNN-Based Wireless Communication Systems." _ACM CCS_, 2023.  
  NetCLR (contrastive pre-training + fine-tuning) implemented in `baselines.py`; trained via `train_baselines.py`.

---

### Defense mechanisms

- Jinjin Gong and Tao Wang. "Zero-delay Lightweight Defenses Against Website Fingerprinting." _USENIX Security Symposium_, 2020.  
  FRONT frontal-padding simulator (`front_defend` in `defenses.py`).

- Marc Juarez, Mohsen Imani, Mike Perry, Claudia Diaz, and Matthew Wright. "Toward an Efficient Website Fingerprinting Defense." _ESORICS_, 2016.  
  WTF-PAD adaptive gap-filling simulator (`wtf_pad_defend` in `defenses.py`).

- C. Hwang, H. Jeon, J. Hong, H. Kang, N. Mathews, G. Kim, and S. E. Oh. "Enhancing Search Privacy on Tor: Advanced Deep Keyword Fingerprinting Attacks and BurstGuard Defense." _ACM ASIA CCS_, 2025.  
  BurstGuard response-burst padding simulator (`burstguard_defend` in `defenses.py`).

---

### iCloud Private Relay

- M. Zohaib, T. Sattarov, J. Müller, and M. Zink. "Is iCloud Private Relay Actually Private?" _IEEE INFOCOM_, 2023.  
  Establishes the passive-observer setting and motivates the threat model used in this thesis.
