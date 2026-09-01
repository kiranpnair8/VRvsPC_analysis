README — EEG-Based User Verification Across PC and VR

This repository contains the complete source code used in our study on EEG-based user authentication and verification across heterogeneous environments (PC and VR). The codebase supports classical machine learning models, deep learning models, and one-class verification pipelines using both REVE embeddings and EEGNet.

The repository is organized to support dataset construction, feature extraction, model training, evaluation, and analysis, with full command-line configurability to reproduce all reported experiments.

1. Repository Structure
.
├── Data/                     # Raw EEG data 
├── Data_processed/           # Preprocessed EEG epochs
├── MATLAB/                   # MATLAB scripts for stat tests
├── models/                   # Pretrained REVE models (reve-base, reve-positions)
├── out/                      # Generated datasets (.npz) and results
├── topoplots/                # KL/MMD topographic visualizations
├── Extras/                   # Helper scripts and utilities and other result csv files
│
├── build_verification_dataset.py  #script to build the dataset
├── train_reve_verifier_eer.py      # main analysis code
├── train_oneclass_reve_verifier.py  # anlysis for one class verification test (SVM/k-NN)
├── mixed_run.py    # helper functions for mixed_train run
├── feature_extractors.py    #feature extraction functions
├── head_utils.py   #ML functions
├── verify_utils.py  #helper functions
├── reve_loader.py    
├── download_reve_once.py   #get the data from reve repo
├── phase1_raweeg.py        #intial stat tests
├── phase2_p300.py          #intial stat tests
├── 
├── verify_utils.py   # function  to label the data
└── test/

2. Core Scripts and Their Purpose
Dataset Construction

build_verification_dataset.py
Builds PC, VR, and mixed verification datasets from preprocessed EEG epochs.
Outputs .npz files containing train/validation/test splits.

Feature Extraction

feature_extractors.py
Implements:

REVE embedding extraction (frozen transformer)

EEGNet feature extraction

Utility functions for batching, embedding, and normalization

download_reve_once.py
Downloads REVE pretrained weights once and stores them locally.

Verification Pipelines
Binary Verification (Genuine vs Impostor)

train_reve_verifier_eer.py
Main script for binary verification experiments using:

REVE + MLP / SVM / KNN / RF

EEGNet (end-to-end)

PC→PC, VR→VR, PC↔VR (cross-environment)

Mixed training (PC+VR)

One-Class Verification

train_oneclass_reve_verifier.py
Implements one-class authentication using:

One-Class SVM

One-Class KNN
Supports within-environment, cross-environment, and mixed training.

Model Utilities

head_utils.py
Training and tuning of classical ML heads (MLP, SVM, KNN, RF).

verify_utils.py
Implements:

EER computation

FAR / FRR at validation threshold

Score aggregation utilities

Analysis & Visualization

topoplots/
Contains code and outputs for KL divergence and MMD topographic maps (16-channel 10–10 system).

MATLAB/
EEG preprocessing, visualization, and sensor-space analyses.

3. Running Experiments
3.1 Binary Verification (REVE / EEGNet)
python train_reve_verifier_eer.py \
  --npz out/pc_vr_verification_dataset.npz \
  --model_dir ./models \
  --model_name SVM \
  --use_claimed_id


Key options

--model_name : MLP | SVM | KNN | RF

--use_claimed_id : Enables claimed-identity conditioning

--cross_env : Enables PC→VR and VR→PC testing

--train_mixed : Train on PC+VR jointly

--tune : Validation-based hyperparameter tuning

--tune_eegnet : Closed-loop EEGNet fine-tuning

3.2 One-Class Verification
python train_oneclass_reve_verifier.py \
  --npz out/pc_vr_oneclass_dataset.npz \
  --model_dir ./models \
  --model ocsvm \
  --env pc \
  --test_env vr


Options

--model : ocsvm | ocknn

--env : Training environment (pc | vr | mixed)

--test_env : Testing environment (pc | vr)

--bs : Batch size for REVE embeddings

4. Hyperparameter Tuning Strategy

All models are trained using closed-loop validation-driven hyperparameter selection, where parameters are optimized to minimize validation Equal Error Rate (EER).

MLP: learning rate, epochs

SVM: C (log-scale)

KNN: number of neighbors

Random Forest: number of trees

EEGNet: learning rate × epochs grid search

The validation threshold is then fixed and applied to the test set to compute EER, FAR, and FRR, ensuring no test leakage.

5. Reproducibility Notes

All random seeds are fixed where applicable

REVE embeddings are frozen

Evaluation strictly follows validation-threshold testing

Dataset splits are subject-independent

6. Contact

For questions or reproduction issues, please contact the authors via the email provided in the paper.