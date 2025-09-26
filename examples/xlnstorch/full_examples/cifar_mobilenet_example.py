import xlnstorch.nn as nn
import xlnstorch as xlt
from xlnstorch.transforms import ToLNSTensor, LNSNormalize
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision import datasets
import argparse
import time
import xlns

# Example precision and base settings (these are mutually exclusive)
parser = argparse.ArgumentParser(description='MobileNet model with LNS')
parser.add_argument('--precision', '-f', type=int, default=None, help='Precision for LNS computations')
parser.add_argument('--base', '-b', type=float, default=None, help='Base for LNS computations')
parser.add_argument('--table', '-t', type=bool, default=False, help='Whether to use table-based LNS computations')
parser.add_argument('--verbose', '-v', type=int, default=10, help='How freq to print batch')
parser.add_argument('--optimizer', '-o', type=str, default='sgd', help='Choose sgd,mul,signmul,hybrid,madam')
parser.add_argument('--learnrate', '-l', type=float, default=0.1, help='Learning rate')
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of Epochs')
args = parser.parse_args()

if args.table:
    if args.precision is None and args.base is None:
        raise ValueError("Must specify precision or base with --table option")
    xlt.set_default_sbdb_implementation("tab")
    xlt.operators.tab.get_table("tmp", f=args.precision, b=args.base)

if args.precision is None and args.base is not None:
    xlns.xlnsB = args.base

elif args.base is None and args.precision is not None:
    xlns.xlnssetF(args.precision)

class DepthwiseSeparableConv(nn.LNSModule):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.LNSSequential(
            nn.LNSConv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                         padding=1, groups=in_channels, bias=False),
            nn.LNSBatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True)
        )
        self.pointwise = nn.LNSSequential(
            nn.LNSConv2d(in_channels, out_channels, kernel_size=1,
                         stride=1, padding=0, bias=False),
            nn.LNSBatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.LNSModule):

    # Depthwise separable conv blocks
    # Each tuple is (out_channels, stride) for the depthwise step
    cfg = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    def __init__(self, num_classes=10, in_channels=3, dropout=0.0):
        super().__init__()

        self.stem = nn.LNSSequential(
            nn.LNSConv2d(in_channels, 32, kernel_size=3, stride=1,
                         padding=1, bias=False),
            nn.LNSBatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )

        layers = []
        in_c = 32
        for out_c, stride in self.cfg:
            layers.append(DepthwiseSeparableConv(in_c, out_c, stride))
            in_c = out_c

        self.features = nn.LNSSequential(*layers)

        self.pool = nn.LNSAdaptiveAvgPool2d(1)
        self.dropout = nn.LNSDropout(dropout)
        self.classifier = nn.LNSLinear(in_c, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(-1, 1024)
        x = self.dropout(x)
        x = self.classifier(x)
        return torch.nn.functional.log_softmax(x, dim=1)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.LNSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LNSBatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LNSLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

device = "cpu"
train_transform = Compose([
    ToLNSTensor(f=args.precision, b=args.base, device=device),
    LNSNormalize(mean=0.5, std=0.5, f=args.precision, b=args.base),
])
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=train_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MobileNetV1().to(device)
loss_func = torch.nn.NLLLoss() # w/ log_softmax, this is equivalent to cross-entropy loss
if args.optimizer=='sgd':
    optimizer = xlt.optim.LNSSGD(model.lns_parameters(), lr=args.learnrate, momentum=0.9)
elif args.optimizer=='mul':
    optimizer = xlt.optim.LNSMul(model.lns_parameters(), lr=args.learnrate, use_pow=False)
elif args.optimizer=='signmul':
    optimizer = xlt.optim.LNSSignMul(model.lns_parameters(), lr=args.learnrate, use_pow=False)
elif args.optimizer=='madam':
    optimizer = xlt.optim.LNSMadam(model.lns_parameters(), lr=args.learnrate, use_pow=False, beta=0.99)
elif args.optimizer=='hybrid':
    optimizer = xlt.optim.LNSHybridMul(model.lns_parameters(), lr=args.learnrate)
    # optimizer = LNSHybridMulAlex(model.lns_parameters(), lr=args.learnrate)

start = time.time()
num_epochs = args.epochs 
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

        if args.verbose > 0:
          if ((i + 1) % args.verbose == 0):
            print(f"Batch {i+1}: {batch_correct}/{target.size(0)} correct.")
            # print(f"Memory usage: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")

    # Calculate average loss and accuracy for the epoch
    train_epoch_loss = running_train_loss / train_total
    train_epoch_acc = train_correct / train_total

    # Validation phase
    model.eval()

    # Track cumulative loss and accuracy for the validation epoch
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Disable gradient calculation for validation
    with torch.no_grad():

        for data, target in test_loader:

            outputs = model(data)
            loss = loss_func(outputs, target)

            # Accumulate the loss and accuracy
            running_val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.value, dim=1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

    # Calculate the average validation loss and accuracy for the epoch
    val_epoch_loss = running_val_loss / val_total
    val_epoch_acc = val_correct / val_total

    # Print epoch summary
    print(f"Epoch {epoch}:")
    print(f"  Training   - Loss = {train_epoch_loss:.4f}, Accuracy = {train_epoch_acc:.4f}")
    print(f"  Validation - Loss = {val_epoch_loss:.4f}, Accuracy = {val_epoch_acc:.4f}")

elapsed = time.time() - start
print(f"Training completed in {elapsed:.2f} seconds.")

# Final test accuracy (optional, as we've already been validating each epoch)
model.eval()
correct = 0
total = 0

# Disable gradient calculation for final evaluation
with torch.no_grad():
    for data, target in test_loader:

        outputs = model(data)
        _, predicted = torch.max(outputs.value, dim=1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

print("\nFinal Test Accuracy:", correct / total)
print("Elapsed time:", elapsed)
