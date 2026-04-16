# Perform mnist classification on the model for sanity check

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visual_encoder import ResNet18


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = ResNet18()
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        features = self.resnet(x)
        out = self.fc(features)
        return out


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Data loading and preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    test_dataset = torch.utils.data.Subset(test_dataset, range(200))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleClassifier().to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            ), labels.to("cuda" if torch.cuda.is_available() else "cpu")

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(
                "cuda" if torch.cuda.is_available() else "cpu"
            ), labels.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
