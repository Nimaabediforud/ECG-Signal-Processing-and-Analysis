# End-to-End ECG Signal Processing and Analysis

**Project Status:** Core signal-processing and QRS detection pipeline implemented. Feature extraction, reusable visualization utilities, automated testing, quantitative evaluation, and multi-recording robustness analysis are still in progress.

## Overview

This project focuses on building an end-to-end ECG signal-processing and analysis pipeline using classical digital signal-processing techniques.

The primary objective is to understand and implement the major stages of ECG processing rather than relying on high-level black-box detection libraries. The project uses recordings from the **MIT-BIH Arrhythmia Database** and is being developed as a modular, reproducible engineering project.

The current implementation covers ECG signal characterization, preprocessing, frequency-domain analysis, and QRS detection, including both offline and sample-by-sample online processing.

## Pipeline
```
ECG Data
   ↓
Signal Acquisition & Representation
   ↓
Signal Visualization
   ↓
Signal Characterization
   ↓
Noise & Artifact Investigation
   ↓
Preprocessing & Filtering
   ↓
Time/Frequency Analysis
   ↓
QRS / R-Peak Detection
   ↓
ECG Feature Extraction
   ↓
Quantitative Evaluation
   ↓
Multi-Recording Validation
   ↓
Robustness & Failure-Case Analysis
   ↓
Modular Final Pipeline
```

## Current Implementation

### ECG Data

The project uses ECG recordings from the **MIT-BIH Arrhythmia Database** accessed through the WFDB Python ecosystem.

The development process includes inspection of:

- Sampling frequency
- Number of channels
- Signal duration
- Signal names
- Signal amplitude characteristics
- Reference annotations
- Time-domain morphology
- Frequency-domain characteristics

> The dataset itself is not intended to be included in the GitHub repository.

### Signal Analysis

Initial signal analysis includes:

- ECG waveform visualization
- Basic statistical characterization
- Time-domain inspection
- Frequency-domain analysis using FFT
- Investigation of relevant frequency components
- Inspection of annotated ECG events
- Investigation of noise and artifacts

### Preprocessing

The preprocessing stage investigates filtering techniques appropriate for ECG signals, including:

- High-pass filtering for baseline-wander reduction
- Band-pass filtering
- Notch filtering for power-line interference
- Zero-phase filtering for offline analysis where appropriate

Filtering choices are investigated rather than treated as universal parameters.

### Pan–Tompkins QRS Detection

A major component of the project is an educational implementation of the **Pan–Tompkins QRS detection algorithm**.

The core processing pipeline is reconstructed from the original algorithm:
```
ECG
 ↓
Band-Pass Filtering
 ↓
Derivative
 ↓
Squaring
 ↓
Moving-Window Integration
 ↓
Candidate Peak Detection
 ↓
Adaptive Thresholding
 ↓
Refractory Period
 ↓
R-Peak Localization
 ↓
Search-Back
```

The implementation is designed to expose and inspect the intermediate processing stages rather than hiding them behind a single library function.

### Implemented Components

The current implementation includes:

- Band-pass filtering
- Derivative operation
- Squaring
- Moving-window integration
- Candidate peak detection
- Adaptive thresholding
- Signal/noise peak classification
- Adaptive threshold updates
- Refractory-period handling
- R-peak localization
- Search-back logic
- State-based sample-by-sample processing

Some components are still being refined and validated against reference annotations.

## Offline and Online Implementations

Two processing approaches are being developed.

### Offline Processing

The offline implementation processes an ECG recording as an available signal sequence and is primarily used for:

- Algorithm development
- Visualization
- Debugging
- Parameter investigation
- Quantitative analysis

### Online / Sample-by-Sample Processing

The online implementation processes the ECG one sample at a time and maintains internal state across samples simulating real-time sampling and signal processing.

This version is intended to reproduce the causal structure of an online ECG detector and provides experience with:

- Recursive filtering
- Stateful processing
- Streaming signal analysis
- Delayed decisions
- Candidate queues
- Adaptive parameters
- Refractory-period logic
- Online R-peak localization

The implementation is described as sample-by-sample processing rather than a clinically validated real-time detector.

## Feature Extraction

**Status: In progress**

Planned ECG features include:

- R-peak locations
- RR intervals
- Heart rate
- Basic HRV-related features
- Additional meaningful time-domain ECG features

Feature extraction will be implemented as a separate modular component after the detection pipeline has been sufficiently validated.

## Quantitative Evaluation

**Status: In progress**

The final evaluation will use the reference annotations provided with the dataset.

Planned evaluation measures include:

- Sensitivity
- Positive Predictive Value (PPV)
- F1-score
- False positives
- False negatives
- R-peak timing error
- Detection error analysis

The evaluation will distinguish between development/debugging recordings and validation recordings to reduce the risk of overfitting parameters to individual ECG records.

## Multi-Recording Validation

**Status: Planned / in progress**

A reliable ECG detector should not be judged from a single recording.

The project therefore aims to evaluate the final pipeline across multiple MIT-BIH recordings using consistent parameters.

The analysis will investigate:

- Generalization across recordings
- Detection performance
- False-positive patterns
- False-negative patterns
- Timing accuracy
- Signal-quality effects
- Failure cases
- Robustness to different ECG morphologies and artifacts

The goal is to understand where the algorithm succeeds and where its assumptions become limiting.

## Visualization

**Status: In progress**

Visualization is treated as part of the engineering and debugging process rather than only as final presentation.

Planned reusable visualization utilities will support inspection of:

- Raw ECG
- Filtered ECG
- Derivative signal
- Squared signal
- Integrated signal
- Candidate peaks
- Signal/noise classification
- Adaptive thresholds
- Refractory-period decisions
- Localized R-peaks
- Reference annotations
- Detection errors
- Feature distributions

## Project Structure
```
ECG-Signal-Processing/
│
├── Data/
│   └── (ECG dataset — excluded from Git)
│
├── Documents/
│   └── Research notes and technical references (Will be added)
│
├── Notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_ecg_signal_characterization.ipynb
│   └── ...
│
├── Results/
│   └── Generated figures and evaluation results (Will be added)
│
├── Src/
│   ├── pan_tompkins_offline.py
│   ├── pan_tompkins_online.py
│   └── ...
│
├── Tests/
│   └── Unit and integration tests (Will be added)
│
├── README.md
├── requirements.txt
└── .gitignore
```

The exact filenames and module organization may evolve as the project is completed.

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- pandas
- WFDB
- Jupyter Notebook

## Engineering Approach

The project follows a sequential development methodology:
```
Learn
  ↓
Implement
  ↓
Inspect
  ↓
Debug
  ↓
Validate
  ↓
Integrate
  ↓
Document
```

Important processing stages are implemented explicitly whenever practical so that their mathematical and signal-processing behavior can be inspected.

Standard algorithms are used as technical references, but the goal is to reconstruct their important components and understand their assumptions rather than simply calling a prebuilt detector.

## Project Status

### Completed

- ECG dataset loading
- Basic signal characterization
- Time-domain analysis
- Frequency-domain analysis
- Initial noise/artifact investigation
- ECG preprocessing and filtering experiments
- Offline QRS detection pipeline
- Sample-by-sample online processing pipeline
- Adaptive thresholding
- Refractory-period handling
- R-peak localization
- Search-back implementation / refinement

### In Progress

- Reusable visualization class
- ECG feature extraction class
- Comprehensive unit tests
- Integration tests
- Annotation-based quantitative evaluation
- Detection error analysis
- Multi-recording validation
- Robustness and failure-case analysis

### Future Extensions

- More advanced ECG features
- Improved modular pipeline
- Performance profiling
- Streaming/real-time experiments
- Potential C implementation of selected algorithms
- Possible embedded deployment

Machine learning and deep learning are intentionally outside the scope of the current classical ECG pipeline. They may be investigated later as a separate extension after the classical processing system has been thoroughly evaluated.

## Limitations

This project is an educational and engineering implementation of classical ECG signal processing.

It should not be interpreted as:

- A clinical diagnostic system
- A medically certified device
- A replacement for professional ECG interpretation
- Evidence of clinical validity

Performance claims will only be made after systematic evaluation against reference annotations across multiple recordings.

## Why This Project?

The project is intended to build a strong foundation in biomedical signal processing by connecting theoretical DSP concepts with a complete real-world biomedical signal-processing problem.

It also provides a foundation for future work in:
```
Classical DSP
     ↓
Biomedical Signal Processing
     ↓
Feature Extraction
     ↓
Robust Signal Analysis
     ↓
Machine Learning
     ↓
Deep Learning
     ↓
Real-Time / Embedded Biomedical Systems
```

## References

The main technical reference for the QRS detection component is:

Pan, J., & Tompkins, W. J. (1985). A Real-Time QRS Detection Algorithm. IEEE Transactions on Biomedical Engineering, BME-32(3), 230–236.

The ECG recordings are obtained from the MIT-BIH Arrhythmia Database through PhysioNet/WFDB.

**Project status: Active development**
