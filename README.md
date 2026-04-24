# Dynamic Evidence Adjustment (DEA)
## Introduction
Dynamic Evidence Adjustment (DEA) is proposed to address the uncertainty quantification issue of Evidential Deep Learning (EDL) on imbalanced datasets. EDL is a widely used paradigm for uncertainty quantification with minimal computational overhead, but it performs poorly on imbalanced data, where minority classes (with fewer samples) suffer from unstable optimization and abnormally high uncertainty even for correctly classified samples, leading to the Reliable Uncertainty Quantification for Imbalanced Data (RUQID) problem.

To solve this problem, DEA integrates two key components: the Adaptive Weight Adjustment (AWA) module dynamically balances the model’s optimization across classes by capturing data-based and evidence-based biases, while the Evidence-guided Memory Calibration Module (ECM) maintains class-specific evidence distributions to calibrate uncertainty, ensuring reliable results for all classes especially minorities. The schematic diagram of the core process of the project is as follows:

![The framework of Dynamic Evidence Adjustment (DEA)](results/DEA.png)

## Dataset
The following six real-world datasets were used to construct imbalanced data in the experiment:

### Data Source
1. [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html): A classic handwritten digit recognition dataset containing 70,000 28×28 grayscale images of 10 digit classes (0–9), widely used for benchmarking image classification models.
2. [Fashion-MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html): A benchmark dataset of 70,000 28×28 grayscale images covering 10 categories of fashion products, designed as a direct replacement for the original MNIST dataset.
3. [CIFAR-10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html): A standard small-image classification dataset with 60,000 32×32 RGB color images distributed across 10 object classes, commonly used for evaluating deep learning models.
4. [SPOTS-10](https://github.com/Amotica/SPOTS-10): A nighttime animal pattern recognition dataset consisting of 10 categories of grayscale images, specifically constructed for low-light and animal detection tasks.
5. [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02): A widely used object recognition dataset with 101 object categories, containing moderate-sized real-world images for general visual recognition research.
6. [Caltech-256](https://data.caltech.edu/records/nyy15-4j048): An extended and improved version of Caltech-101, featuring 256 object categories with more images and greater intra-class variation.

### Data References
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.
- Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.
- Atanbori, J. (2024). SPOTS-10: Animal Pattern Benchmark Dataset for Machine Learning Algorithms.
- Fei-Fei, L., Fergus, R., & Perona, P. (2004, June). Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories.
- Griffin, G., Holub, A., & Perona, P. (2007). Caltech-256 object category dataset.

## Project Structure
The following is a description of the core directory of the DEA project. All directories and files are classified by function for easy reference and maintenance:
```bash
DEA/                                     # Project root directory
├── data_processing/                     # Data preprocessing & dataset construction scripts
│   ├── construct-[dataset_name].py      # Build imbalanced datasets for specified dataset
│   ├── pre-training-[model_name].py     # Extract image features using frozen pre-trained models
│   ├── count-train.py                   # Count sample numbers per class in training set
│   └── dc.py                            # Calculate inherent classification difficulty (dc)
├── datasets/                            # Dataset storage directory
│   ├── [dataset_name]/                  # Raw & processed datasets
│   ├── count_train.json                 # Class-wise sample counts of training data
│   └── Dc_[model_name].json             # Precomputed inherent classification difficulty scores
├── metrics/                             # Evaluation metrics scripts
│   ├── acc_auprc.py                     # Compute classification accuracy and AU-PRC
│   ├── ece_class.py                     # Calculate class-wise Expected Calibration Error (ECE)
│   ├── ece_overall.py                   # Calculate overall model Expected Calibration Error (ECE)
│   └── ood_auprc_auroc.py               # Evaluate OOD detection: AU-ROC & AU-PRC
├── results/                             # Experimental results & inference outputs
├── requirements.txt                     # Project dependencies and environment configuration
├── dataset_reader.py                    # Dataset loading module
├── loss_funcation.py                    # Custom loss functions for model training
├── main.py                              # Main entry: model training, validation, and testing
├── model.py                             # Model architecture and network definition
└── README.md                            # Project documentation and user guide
```
To run the project, follow the steps below in order:
1. Obtain the required datasets and place them in the corresponding subdirectories under the `datasets/` folder.
2. Sequentially run the scripts in the `data_processing/` directory to construct imbalanced datasets.
3. Execute `python3 main.py` in the project root directory to start model training and testing.
