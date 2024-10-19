import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.5):
        super(MyModel, self).__init__()
        
        # Define CNN architecture using nn.Sequential
        self.model = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: (3, 224, 224), Output: (32, 224, 224)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (64, 112, 112)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: (128, 56, 56)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: (256, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 14, 14)
            
            # Fifth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Output: (512, 14, 14)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 7, 7)
            
            # Flatten
            nn.Flatten(),  # Output: (512 * 7 * 7)
            
            # Fully connected layers
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            
            nn.Linear(512, num_classes)  # Output: (num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the network
        return self.model(x)



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
