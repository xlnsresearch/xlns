import xlnstorch as xlt
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import xlnstorch as xltorch
from xlnstorch.transforms import ToLNSTensor
import xlns

# Parse command line arguments
parser = argparse.ArgumentParser(description='MNIST training on small CNN with LNS')
parser.add_argument('--precision', '-f', type=int, default=None, help='Precision for LNS computations')
parser.add_argument('--base', '-b', type=float, default=None, help='Base for LNS computations')
parser.add_argument('--table', '-t', type=bool, default=False, help='Whether to use table-based LNS computations')
args = parser.parse_args()

if args.table:
    if args.precision is None and args.base is None:
        raise ValueError("Must specify precision or base with --table option")
    xltorch.set_default_sbdb_implementation("tab")
    xltorch.operators.tab.get_table("tmp", f=args.precision, b=args.base)

if args.precision is None and args.base is not None:
    xlns.xlnsB = args.base

elif args.base is None and args.precision is not None:
    xlns.xlnssetF(args.precision)

class LNSTinyCNN(xltorch.nn.LNSModule):

    def __init__(self, out1=16, out2=32, out_mp2=7*7):
        super().__init__()
        self.conv1 = xltorch.nn.LNSConv2d(1, out1, kernel_size=5, padding=2)
        self.mp1 = xltorch.nn.LNSMaxPool2d(kernel_size=2, stride=2)
        self.conv2 = xltorch.nn.LNSConv2d(out1, out2, kernel_size=5, padding=2)
        self.mp2 = xltorch.nn.LNSMaxPool2d(kernel_size=2, stride=2)
        self.fc = xltorch.nn.LNSLinear(out2 * out_mp2, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)

device = "cpu"
train_transform = ToLNSTensor(f=args.precision, b=args.base, device=device)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=train_transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LNSTinyCNN().to(device)
loss_func = torch.nn.NLLLoss() # w/ log_softmax, this is equivalent to cross-entropy loss
optimizer = xltorch.optim.LNSSGD(model.lns_parameters(), lr=0.1, momentum=0.9)

start = time.time()
num_epochs = 15
for epoch in range(1, num_epochs + 1):

    model.train()

    # Track cumulative loss and accuracy for the training epoch
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0
    t1 = time.time()

    for i, (data, target) in enumerate(train_loader):
        batch_t1 = time.time()
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
        batch_t2 = time.time()

        print(f"Batch {i+1} ({(batch_t2 - batch_t1):.2f}s): {batch_correct}/{target.size(0)} correct.")
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
    t2 = time.time()

    # Print epoch summary
    print(f"Epoch {epoch} ({(t2 - t1):.2f}s):")
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
