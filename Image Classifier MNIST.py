import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Accuracy, Precision, Recall
from typing import Tuple, List

# =============================== DATASET LOADING ===============================

def load_fashion_mnist(batch_size_train: int, batch_size_test: int) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the FashionMNIST dataset and returns train and test dataloaders.
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

# =============================== MODEL ===============================

class CNN(nn.Module):
    """
    A simple CNN for FashionMNIST classification.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 14, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(14, 28, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(28 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# =============================== TRAINING ===============================

def train_model(model: nn.Module, dataloader: DataLoader, optimizer, criterion, epochs: int) -> None:
    """
    Trains the model using the provided data loader, optimizer, and loss function.
    """
    model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# =============================== EVALUATION ===============================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    metric_accuracy: Accuracy,
    metric_precision: Precision,
    metric_recall: Recall
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model and returns accuracy, per-class precision, and recall.
    """
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            metric_accuracy(preds, labels)
            metric_precision(preds, labels)
            metric_recall(preds, labels)

    accuracy = metric_accuracy.compute()
    precision = metric_precision.compute()
    recall = metric_recall.compute()

    return accuracy, precision, recall

# =============================== METRIC RESET ===============================

def reset_metrics(metrics: List) -> None:
    """
    Resets all provided metric objects.
    """
    for metric in metrics:
        metric.reset()

# =============================== ORCHESTRATION ===============================

def run_experiments(
    learning_rates: List[float],
    epochs: int,
    num_classes: int = 10,
    batch_size_train: int = 2,
    batch_size_test: int = 256
) -> None:
    """
    Runs training and evaluation for each learning rate.
    """
    train_loader, test_loader = load_fashion_mnist(batch_size_train, batch_size_test)

    metric_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    metric_precision = Precision(task="multiclass", num_classes=num_classes, average=None)
    metric_recall = Recall(task="multiclass", num_classes=num_classes, average=None)

    for lr in learning_rates:
        print(f"\n=== Training with Learning Rate: {lr} ===")
        model = CNN(num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        reset_metrics([metric_accuracy, metric_precision, metric_recall])

        train_model(model, train_loader, optimizer, criterion, epochs)
        accuracy, precision, recall = evaluate_model(model, test_loader, metric_accuracy, metric_precision, metric_recall)

        print("Accuracy:", accuracy.item())
        print("Precision (per class):", precision.tolist())
        print("Recall (per class):", recall.tolist())

# =============================== ENTRY POINT ===============================

if __name__ == "__main__":
    run_experiments(
        learning_rates=[0.00001, 0.00005, 0.0001,0.0005, 0.001,0.05, 0.01, 0.05, 0.1, 0.5],
        epochs=10
    )

