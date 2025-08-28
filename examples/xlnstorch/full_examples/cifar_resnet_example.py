"""Example modified from https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/"""
import xlnstorch.nn as nn
import xlnstorch as xlt
from xlnstorch.transforms import ToLNSTensor, LNSNormalize
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from torchvision import datasets
import argparse
import time
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='CIFAR training with LNS')
parser.add_argument('--precision', '-f', type=int, default=None, help='Precision for LNS computations')
parser.add_argument('--base', '-b', type=float, default=None, help='Base for LNS computations')
parser.add_argument('--table', '-t', type=bool, default=False, help='Whether to use table-based LNS computations')
args = parser.parse_args()

if args.table:
    if args.precision is None and args.base is None:
        raise ValueError("Must specify precision or base with --table option")
    xlt.set_default_sbdb_implementation("tab")
    xlt.operators.implementations.tab.get_table("tmp", f=args.precision, b=args.base)

class BasicBlock(nn.LNSModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.LNSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.LNSBatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = nn.LNSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.LNSBatchNorm2d(out_channels)

        self.shortcut = nn.LNSSequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.LNSSequential(
                nn.LNSConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.LNSBatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.LNSModule):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.LNSConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.LNSBatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = nn.LNSMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.LNSAdaptiveAvgPool2d((1, 1))
        self.fc = nn.LNSLinear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.LNSSequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return torch.nn.functional.log_softmax(out, dim=1)

def stratified_indices(dataset, n_per_class, seed=42):
    rng = np.random.default_rng(seed)

    # CIFAR10 exposes labels in `targets` (older versions may use `train_labels`/`test_labels`)
    targets = getattr(dataset, 'targets', None)
    if targets is None:
        targets = getattr(dataset, 'train_labels', getattr(dataset, 'test_labels', None))
    targets = np.array(targets)

    inds = []
    for c in np.unique(targets):
        cls_inds = np.where(targets == c)[0]
        rng.shuffle(cls_inds)
        inds.extend(cls_inds[:n_per_class].tolist())

    rng.shuffle(inds)
    return inds

device = "cpu"
train_transform = Compose([
    ToLNSTensor(f=args.precision, b=args.base, device=device),
    LNSNormalize(mean=0.5, std=0.5, f=args.precision, b=args.base),
])
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)

batch_size = 5
num_per_class = 2

train_inds = stratified_indices(train_dataset, num_per_class)
train_subset = Subset(train_dataset, train_inds)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

model = ResNet18(10).to(device)
loss_func = torch.nn.NLLLoss() # w/ log_softmax, this is equivalent to cross-entropy loss
optimizer = xlt.optim.LNSAdam(model.lns_parameters(), lr=0.001, betas=(0.9, 0.99))

print(f"Training with {num_per_class} samples per class (10 classes total). No validation set.")

start = time.time()
num_epochs = 20
for epoch in range(1, num_epochs + 1):

    model.train()

    # Track cumulative loss and accuracy for the training epoch
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = loss_func(outputs, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_train_loss += loss.value.item() * data.size(0)
        _, predicted = torch.max(outputs.value, dim=1)
        train_total += target.size(0)
        batch_correct = (predicted == target).sum().item()
        train_correct += batch_correct

        print(f"Batch {i+1}: {batch_correct}/{target.size(0)} correct.")

    # Calculate average loss and accuracy for the epoch
    train_epoch_loss = running_train_loss / train_total
    train_epoch_acc = train_correct / train_total

    # Print epoch summary
    print(f"Epoch {epoch}:")
    print(f"  Training   - Loss = {train_epoch_loss:.4f}, Accuracy = {train_epoch_acc:.4f}")

elapsed = time.time() - start
print(f"Training completed in {elapsed:.2f} seconds.")