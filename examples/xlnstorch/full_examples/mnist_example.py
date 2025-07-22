import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import xlnstorch as xltorch
from xlnstorch.transforms import ToLNSTensor

class LNSNet(xltorch.nn.LNSModule):

    def __init__(self):
        super().__init__()
        self.fc1 = xltorch.nn.LNSLinear(784, 100)
        self.fc2 = xltorch.nn.LNSLinear(100, 10)

        # Initialize the weights and biases of the linear layers
        # with normal distribution for weights and zeros for biases.
        xltorch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        xltorch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        xltorch.nn.init.zeros_(self.fc1.bias)
        xltorch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 784)
        # Apply the linear layers with ReLU activation
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        # Apply log softmax to the output (to get log probabilities)s
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

# Set up MNIST datasets with basic transforms (converting images to tensors)
device = "cpu"
train_transform = ToLNSTensor(device=device)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=train_transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LNSNet().to(device)
loss_func = torch.nn.NLLLoss() # w/ log_softmax, this is equivalent to cross-entropy loss
optimizer = xltorch.optim.LNSSGD(model.parameter_groups(), lr=0.1, momentum=0.9)

start = time.time()
num_epochs = 5
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

        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}: {batch_correct}/{target.size(0)} correct.")

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