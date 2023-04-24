"""This module contains machine learning functionalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x


def train_model(
    x_train: pd.DataFrame, y_train: pd.DataFrame, num_epochs: int
) -> NeuralNet:
    """Train the PyTorch model.

    Args:
        x_train: Training data.
        y_train: Training labels.
        num_epochs: Number of epochs.

    Returns:
        model: Trained model.
    """

    print("ml.train_model: Training machine learning model...")

    model = NeuralNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)

    for epoch in range(num_epochs + 1):
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            correct = 0
            for i in range(len(y_pred)):
                if torch.argmax(y_pred[i]) == y_train[i]:
                    correct += 1
            accuracy = correct / len(y_pred)
            print(
                "ml.train_model: Epoch: ",
                epoch,
                "| Accuracy: ",
                accuracy,
                "| Loss: ",
                loss.item(),
            )

    return model


def evaluate_model(x_test: pd.DataFrame, y_test: pd.DataFrame, model: NeuralNet):
    """Evaluate the PyTorch model.

    Args:
        x_test: Test data.
        y_test: Test labels.
        model: Trained model.
    """

    x_test = torch.tensor(x_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    with torch.no_grad():
        y_pred = model(x_test)
        correct = 0
        for i in range(len(y_pred)):
            if torch.argmax(y_pred[i]) == y_test[i]:
                correct += 1
        accuracy = correct / len(y_pred)
        print("ml.evaluate_model: Test Accuracy: ", accuracy)


def predict(x: list, model: NeuralNet) -> tuple:
    """Predict the learning style of a user.

    Args:
        x: The user data as array.
        model: The trained model.

    Returns:
        The predicted learning style group ID and confidence.
    """

    learning_styles = {
        1: "sensing",
        2: "intuitive",
        3: "visual",
        4: "verbal",
        5: "active",
        6: "reflective",
        7: "sequential",
    }

    user_id = x[0]
    x = x[1:]

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x)
        label = torch.argmax(y_pred)

    confidence = round(y_pred[0][label.item()].item(), 2)

    print(
        "ml.predict: User ID: ",
        user_id,
        "| Learning Style: ",
        learning_styles[label.item() + 1],
        "| Confidence: ",
        confidence,
    )

    return label.item() + 1, confidence


def save_model(model: NeuralNet, path: str):
    """Save the PyTorch model.

    Args:
        model: The trained model.
        path: The path to save the model.
    """

    torch.save(model.state_dict(), path)

    print("ml.save_model: Model saved to {}.".format(path))


def load_model(path: str) -> NeuralNet:
    """Load the PyTorch model.

    Args:
        path: The Path to the model.

    Returns:
        The loaded model.
    """

    model = NeuralNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    print("ml.load_model: Model loaded from {}.".format(path))

    return model
