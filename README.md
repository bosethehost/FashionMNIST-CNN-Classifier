# FashionMNIST CNN Classification
📌 Overview

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the FashionMNIST dataset.
It explores the effect of different learning rates on model performance using metrics such as accuracy, precision, and recall.

The pipeline covers:

i) Loading the FashionMNIST dataset
ii) Building a simple CNN architecture
iii) Training the model with configurable hyperparameters
iv) Evaluating the model using multiple performance metrics
v) Running experiments across various learning rates


🛠 Features

i) Custom CNN with two convolutional layers and max pooling
ii) Uses Adam optimizer and CrossEntropyLoss
iii) Built-in evaluation metrics from torchmetrics

Experiment runner to compare different learning rates

Prints accuracy, per-class precision, and per-class recall


📦 Requirements

Install dependencies via:

pip install torch torchvision torchmetrics

▶️ How to Run
python main.py


By default, it runs experiments with the following learning rates:

[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.05, 0.01, 0.05, 0.1, 0.5]


and trains for 10 epochs each.

⚙️ Customization

You can modify:

i) Learning rates: in run_experiments()
ii) Epochs: in run_experiments()
iii) Batch sizes: batch_size_train and batch_size_test
iv) Number of classes (default: 10 for FashionMNIST)

Example:

if __name__ == "__main__":
    run_experiments(
        learning_rates=[0.001, 0.01],
        epochs=5,
        batch_size_train=64,
        batch_size_test=256
    )

📊 Output Example
=== Training with Learning Rate: 0.001 ===
Accuracy: 0.875
Precision (per class): [0.85, 0.88, 0.89, ...]
Recall (per class): [0.87, 0.85, 0.88, ...]

📈 Potential Improvements

i) Add GPU support for faster training
ii) Implement data augmentation for better generalization
iii) Plot accuracy and loss curves over epochs
iv) Save trained models for later inference

📝 License

This project is open-source and available under the MIT License.
