# Enhanced Drone Classification using Transfer Learning and Optimized RF-Spectrogram

Code and outputs for the paper **"Enhanced Drone Classification using Transfer Learning and Optimized RF-Spectrogram"**.

**Authors:** Kaushik A R, Annaamalai U, Padmavathi S  
**Affiliation:** Department of Information Technology, Thiagarajar College of Engineering, Madurai, Tamilnadu, India

---

## Overview

This repository implements a drone classification system that treats RF (radio-frequency) signal classification as a computer vision task. Raw IQ data are converted into optimized spectrogram images and classified using transfer learning with pre-trained convolutional neural networks. The work contributes (i) an enhanced processing pipeline from RF signals to spectrogram images, and (ii) a systematic evaluation of nine CNN architectures ranging from lightweight to heavier models.

---

## Dataset

The work uses the RF drone dataset by Gluge et al. [[1]](#reference), containing signals from six drone models and remote controllers. The dataset has **seven classes**: DJI, FutabaT14, FutabaT7, Graupner, Noise, Taranis, and Turnigy. Data are provided as IQ samples; this repo includes a pipeline to convert them into spectrogram images for image-based classification.

---

## Methods and Approaches

### Enhanced Data Processing Pipeline

The pipeline converts raw RF data into standardized image inputs for CNNs:

1. **IQ → Spectrogram:** Complex IQ signals are transformed into 2D spectrograms (e.g., via FFT / `torchaudio.transforms.Spectrogram`).
2. **Power spectrum:** Magnitude is computed from real and imaginary components; log scaling (e.g., log₁₀) is applied to improve visibility of features.
3. **Normalization:** Values are brought to a standard range (e.g., min–max normalization).
4. **Image export:** Grayscale power-spectrum images are saved (e.g., PNG) and later resized to fixed dimensions (224×224 for most models, 299×299 for Inception-v3).
5. **Input preparation:** Images are channelized to 3 channels and normalized with **ImageNet** statistics so that pre-trained ImageNet models can be used with transfer learning.

This design reframes the problem as image classification and enables the use of established image preprocessing and transfer learning.

### Model Architectures

Nine CNN architectures are evaluated, grouped by computational profile:

| Category | Models |
|----------|--------|
| **Lightweight** | SqueezeNet, ShuffleNet, MobileNetV2 |
| **Balance** | EfficientNet-B0, DenseNet-121, ResNet-50 |
| **Heavier** | ResNeXt-50, Wide ResNet-50, Inception-v3 |

All models are initialized with **ImageNet pre-trained weights** and fine-tuned for the seven-class drone RF classification task.

### Experimental Setup

- **Data split:** Stratified sampling into training (70%), validation (15%), and test (15%) to preserve class distribution.
- **Optimizer:** Adam with initial learning rate 1e-4.
- **Loss:** Cross-entropy.
- **Scheduler:** ReduceLROnPlateau (e.g., reduce learning rate by factor 0.1 after a fixed number of epochs without validation loss improvement).
- **Training:** Batch size 16, up to 50 epochs, with early stopping (e.g., after 5 epochs without improvement).

### Evaluation

Models are assessed using:

- Classification accuracy (and related metrics)
- Model size (parameters, saved checkpoint size)
- Inference time per sample
- Training time
- Computational cost: FLOPS and MACs (via e.g. `thop`)
- Confusion matrices
- ROC curves and related metrics
- Training/validation curves

No result values are reported in this README; see the paper and the saved metrics/plots in the repo for outcomes.

---

## Repository Structure

```
.
├── README.md
├── Data extraction/
│   └── drone-data.ipynb          # IQ → spectrogram → PNG pipeline; dataset creation
├── ResNet,MobileNet,InceptionV3/
│   ├── dronerf-res-mobilenet-inception-v3.ipynb
│   └── results/                  # Checkpoints, metrics, plots for ResNet-50, MobileNetV2, Inception-v3
├── ShuffleNet,SqueezeNet,EfficientNet/
│   ├── dronerf-shuffle-squeeze-efficientnet.ipynb
│   └── results/                  # Checkpoints, metrics, plots for SqueezeNet, ShuffleNet, EfficientNet-B0
├── DenseNet,ResneXt-50,Wide Resnet-50/
│   ├── dronerf-densenet-resnext-50-wide-resnet-50.ipynb
│   └── results/                  # Checkpoints, metrics, plots for DenseNet-121, ResNeXt-50, Wide ResNet-50
├── visualizations/
│   ├── dronerf-visualizations.ipynb   # Sample spectrogram plots, model comparison figures
│   └── results/
└── paper-diagrams/               # Figures for the paper (framework, sample plots, comparisons)
```

- **Data extraction:** Builds the spectrogram image dataset from IQ data.
- **Model notebooks:** Each notebook trains and evaluates a subset of the nine CNNs and writes metrics, confusion matrices, ROC curves, and training history to the corresponding `results/` folder.
- **Visualizations:** Produces sample spectrograms (e.g., FutabaT14) and model comparison plots (e.g., bar and radar).
- **paper-diagrams:** Contains selected figures used in the paper.

---

## Dependencies

The notebooks use standard Python deep-learning and signal-processing libraries, including:

- **PyTorch** and **torchvision** (models, training, ImageFolder, transforms)
- **torchaudio** (e.g., `Spectrogram` in the data pipeline)
- **scikit-learn** (stratified splits, metrics, confusion matrix, ROC)
- **thop** (FLOPs and MACs)
- **matplotlib**, **numpy**, **pandas**, **tqdm**

Paths in the notebooks may assume a specific layout (e.g., Kaggle inputs or a local `clean_spectrograms/` directory). Adjust paths to match your environment and dataset location.

---

## Reference

[1] Gluge et al., dataset containing RF signals from six drone models and remote controllers (see paper for full citation).

---

## License

See the repository or paper for license and usage terms.
