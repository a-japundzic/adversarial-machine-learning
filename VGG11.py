import torch.nn as nn


# Put this in its own file for organization and it looks cool setup like this
# For this I just followed the assignment and the paper specifications

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        # VGG11 architecture as described in the paper/assignment
        # Convolutional layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1,   64, 3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(256,512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(512,512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
        )

        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        # Pass through the convolutional layers
        x = self.convolutional_layers(x)
        # Flatten the output, needed to pass through the fully connected layers
        x = self.flatten(x)
        # Pass through the fully connected layers
        x = self.fully_connected_layers(x)

        return x

