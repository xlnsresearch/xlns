import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import xlnstorch as xltorch
from xlnstorch.transforms import ToLNSTensor

xltorch.operators.set_default_sbdb_implementation("tab")
xltorch.operators.implementations.tab.get_table("tmp", f=8)

class LNSNet(xltorch.nn.LNSModule):

    def __init__(self):
        super().__init__()
        self.conv = xltorch.nn.LNSConv2d(1, 32, kernel_size=5, weight_f=8, bias_f=8)
        self.fc = xltorch.nn.LNSLinear(32 * 24 * 24, 10, weight_f=8, bias_f=8) # After conv layer, the input size is 32x24x24

        # Initialize the weights and biases of the linear layers
        # with normal distribution for weights and zeros for biases.
        xltorch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.1)
        xltorch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Apply the convolutional layer with ReLU activation
        x = torch.nn.functional.relu(self.conv(x))

        # Flatten the output of convolutional layers
        x = x.view(x.size(0), -1) # flatten to (batch_size, 32 * 7 * 7)

        # Apply the linear layer with ReLU activation
        x = torch.nn.functional.relu(self.fc(x))

        # Apply log softmax to the output (to get log probabilities)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

# Set up MNIST datasets with basic transforms (converting images to tensors)
train_transform = ToLNSTensor(f=8)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=train_transform)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cpu"
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

        # Convert data and target to the appropriate device
        data, target = data.to(device), target.to(device)

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

        # if (i + 1) % 10 == 0:
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

            data, target = data.to(device), target.to(device)

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

        data, target = data.to(device), target.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.value, dim=1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

print("\nFinal Test Accuracy:", correct / total)
print("Elapsed time:", elapsed)