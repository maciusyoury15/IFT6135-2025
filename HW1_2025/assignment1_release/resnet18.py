'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.avg_pool2d(out, 4)  # Global Average Pooling (GAP)
        out = torch.flatten(out, 1)  # Flatten before Linear Layer
        logits = self.linear(out)
        return logits


    def visualize(self, logdir: str, layer_name: str='conv1'):
        """Visualize the kernels of a convolutional layer."""
        
        # Find the layer by its name
        layer = dict(self.named_modules()).get(layer_name, None)

        # Ensure the layer exists and has weight data
        if layer is None or not isinstance(layer, nn.Conv2d):
            print(f"Layer '{layer_name}' not found or is not a Conv2d layer.")
            return

        # Extract the weights (kernels) from the layer
        kernels = layer.weight.data.cpu().numpy()

        # Normalize the kernel values between 0 and 1
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

        # Get the number of kernels and kernel size
        num_kernels, in_channels, kernel_h, kernel_w = kernels.shape

        # Determine grid size for visualization
        grid_size = math.ceil(math.sqrt(num_kernels))

        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.suptitle(f"Kernels of {layer_name}", fontsize=14)

        for i, ax in enumerate(axes.flat):
            if i < num_kernels:
                if in_channels == 3:
                    # Convert 3-channel kernel into an RGB image
                    kernel_img = np.transpose(kernels[i], (1, 2, 0))  # (H, W, C)
                    ax.imshow(kernel_img)
                else:
                    # Grayscale visualization for single-channel kernels
                    ax.imshow(kernels[i, 0], cmap='gray')
                ax.set_title(f"Kernel {i}")
                ax.axis('off')
            else:
                ax.axis('off')  # Hide empty subplots

        # Save the figure
        os.makedirs(logdir, exist_ok=True)
        save_path = os.path.join(logdir, f"Resnet18_{layer_name}_kernels.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Kernel visualization saved to {save_path}")