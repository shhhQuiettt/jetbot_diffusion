import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.flop_counter import FlopCounterMode


class SpatialSoftmax(nn.Module):
    """
    Computes the spatial softmax pooling, returning the expected (x, y) coordinates
    for each channel. Output shape is (B, C * 2).
    """

    def __init__(self, temperature=1.0, learnable_temp=True):
        super().__init__()
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

    def forward(self, x):
        b, c, h, w = x.shape

        # Flatten spatial dimensions and apply softmax with temperature
        x_flat = x.view(b, c, h * w)
        weights = F.softmax(x_flat / self.temperature, dim=-1)

        # Generate normalized grid [-1, 1] on the fly to support dynamic image sizes
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        # Compute expected coordinates: sum(probability * coordinate)
        expected_x = torch.sum(weights * grid_x, dim=-1)
        expected_y = torch.sum(weights * grid_y, dim=-1)

        # Stack coordinates into an expected feature vector [B, C * 2]
        return torch.cat([expected_x, expected_y], dim=-1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        super().__init__()
        # Ensure we don't request more groups than channels
        groups1 = min(num_groups, out_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(groups1, out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(groups1, out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            groups_down = min(num_groups, out_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups_down, out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, input_channels=3, embedding_size=128, num_groups=16):
        super().__init__()
        self.in_channels = 64
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(
            input_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(
            min(self.num_groups, self.in_channels), self.in_channels
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.spatial_softmax = SpatialSoftmax()

        self.fc1= nn.Linear(512 * 2, 512)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, embedding_size)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(
            BasicBlock(self.in_channels, out_channels, stride, self.num_groups)
        )
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(
                    self.in_channels, out_channels, stride=1, num_groups=self.num_groups
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Base feature extraction
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Shape: [B, 512, H_final, W_final]

        x = self.spatial_softmax(x)  # Shape: [B, 1024]
        x = self.fc1(x)  
        x = self.relu_fc(x)
        embedding = self.fc2(x)  # Shape: [B, embedding_size]

        return embedding


# --- Example Usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(input_channels=3, embedding_size=128).to(device)
    print("Model Summary:")
    summary(model, (3, 224, 224))

    with FlopCounterMode(display=True) as flop_counter:
        dummy_input = torch.randn(4, 3, 224, 224).to(device)
        output = model(dummy_input)
