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

def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    assert c % groups == 0, "Channels must be divisible by groups."
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x

class ShuffleV1Block(nn.LNSModule):

    def __init__(self, in_c, out_c, stride, groups = 3):
        super().__init__()
        self.stride = stride
        self.groups = groups

        if stride == 1:
            assert in_c == out_c, "For stride-1 the in/out channels must match."
            branch_out = in_c // 2
            in_channels_branch = branch_out
        else:
            branch_out = out_c - in_c
            in_channels_branch = in_c

        # 1 × 1 group conv
        self.group1 = nn.LNSSequential(
            nn.LNSConv2d(in_channels_branch, branch_out,
                         1, 1, 0, groups=groups, bias=False),
            nn.LNSBatchNorm2d(branch_out),
            torch.nn.ReLU()
        )

        # 3 × 3 dw-conv
        self.dwconv = nn.LNSSequential(
            nn.LNSConv2d(branch_out, branch_out, 3, stride, 1,
                         groups=branch_out, bias=False),
            nn.LNSBatchNorm2d(branch_out),
        )

        # 1 × 1 group conv
        self.group2 = nn.LNSSequential(
            nn.LNSConv2d(branch_out, branch_out, 1, 1, 0,
                         groups=groups, bias=False),
            nn.LNSBatchNorm2d(branch_out),
            torch.nn.ReLU()
        )

        self.avgpool = nn.LNSAvgPool2d(3, stride=2, padding=1) if stride == 2 else None

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = self.group2(self.dwconv(self.group1(x2)))
            out = torch.cat((x1, out), dim=1)

        else:
            out = self.group2(self.dwconv(self.group1(x)))
            out = torch.cat((self.avgpool(x), out), dim=1)

        out = channel_shuffle(out, groups=2)
        return out

class ShuffleNetV1(nn.LNSModule):

    stage_repeats = [3, 7, 3]
    out_channels = [24, 240, 480, 960]

    def __init__(self, num_classes = 10, in_channels = 3, groups = 3, dropout = 0.0):
        super().__init__()
        self.groups = groups

        self.conv1 = nn.LNSSequential(
            nn.LNSConv2d(in_channels, self.out_channels[0],
                         3, 2, 1, bias=False),
            nn.LNSBatchNorm2d(self.out_channels[0]),
            torch.nn.ReLU(inplace=True)
        )
        self.maxpool = nn.LNSMaxPool2d(3, 2, 1)

        in_c = self.out_channels[0]
        blocks = []
        for repeats, out_c in zip(self.stage_repeats, self.out_channels[1:]):
            blocks.append(ShuffleV1Block(in_c, out_c, stride=2,
                                         groups=self.groups))
            in_c = out_c

            for _ in range(repeats - 1):
                blocks.append(ShuffleV1Block(in_c, out_c, stride=1,
                                             groups=self.groups))
        self.stages = nn.LNSSequential(*blocks)

        self.pool = nn.LNSAdaptiveAvgPool2d(1)
        self.dropout = nn.LNSDropout(dropout)
        self.classifier = nn.LNSLinear(in_c, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
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

model = ShuffleNetV1().to(device)
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
