"""This module contains machine learning functionalities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def train_model(x_train, y_train, num_epochs):
    """Train the PyTorch model.

    Args:
        x_train: Training data.
        y_train: Training labels.
        num_epochs: Number of epochs.

    Returns:
        model: Trained model.
    """

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

    model = NeuralNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_train = torch.tensor(x_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)

    for epoch in range(num_epochs+1):
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
            print("Epoch: ", epoch, "| Accuracy: ", accuracy, "| Loss: ", loss.item())

    return model


def evaluate_model(x_test, y_test, model):
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
        print("Test Accuracy: ", accuracy)


def predict(x, model):
    """Predict the learning style of a student.

    Args:
        x: Student data.
        model: Trained model.

    Returns:
        The predicted learning style group ID.
    """

    x = torch.tensor(x.values, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(x)
        label = torch.argmax(y_pred)

    print(
        "Label: ",
        label.item(),
        "| Confidence: ",
        round(y_pred[0][label.item()].item() * 100, 2),
        "%",
    )

    return label.item() + 1


def save_model(model, path):
    """Save the PyTorch model.

    Args:
        model: Trained model.
        path: Path to save the model.
    """

    print("Saving model to: ", path)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    torch.save(model.state_dict(), path)


def load_model(path):
    """Load the PyTorch model.

    Args:
        path: Path to the model.

    Returns:
        The loaded model.
    """

    model = NeuralNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    print("Model loaded from: ", path)
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    return model
