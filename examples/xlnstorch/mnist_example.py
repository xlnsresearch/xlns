import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import xlnstorch as xltorch

class LNSNet(xltorch.layers.LNSModule):

    def __init__(self):
        super().__init__()
        self.fc1 = xltorch.layers.LNSLinear(784, 100)
        self.fc2 = xltorch.layers.LNSLinear(100, 10)

        # Initialize the weights and biases of the linear layers
        # with normal distribution for weights and zeros for biases.
        # This will be made easier with LNSTensor, but it hasn't
        # been implemented yet.
        weight1, bias1 = torch.empty(100, 784), torch.empty(100)
        weight2, bias2 = torch.empty(10, 100), torch.empty(10)

        torch.nn.init.normal_(weight1, mean=0.0, std=0.1)
        torch.nn.init.normal_(weight2, mean=0.0, std=0.1)
        torch.nn.init.zeros_(bias1)
        torch.nn.init.zeros_(bias2)

        self.fc1.weight_lns.data.copy_(xltorch.lnstensor(weight1)._lns)
        self.fc2.weight_lns.data.copy_(xltorch.lnstensor(weight2)._lns)
        self.fc1.bias_lns.data.copy_(xltorch.lnstensor(bias1)._lns)
        self.fc2.bias_lns.data.copy_(xltorch.lnstensor(bias2)._lns)

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
train_transform = transforms.ToTensor()
raw_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
raw_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=train_transform)

# Convert the datasets into in-memory tensors as they are loaded on the fly by default.
# Note: raw_train_dataset.data is of shape [60000, 28, 28] and is of type torch.uint8.
# We unsqueeze to add a channel dimension and convert to float, scaling to [0,1].
train_data = raw_train_dataset.data.unsqueeze(1).float() / 255.0
train_targets = raw_train_dataset.targets
test_data = raw_test_dataset.data.unsqueeze(1).float() / 255.0
test_targets = raw_test_dataset.targets

# Create TensorDatasets and DataLoaders based on the in-memory data.
train_dataset = TensorDataset(train_data, train_targets)
test_dataset = TensorDataset(test_data, test_targets)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cpu"
model = LNSNet().to(device)
loss_func = torch.nn.NLLLoss() # w/ log_softmax, this is equivalent to cross-entropy loss
optimizer = xltorch.optimizers.LNSSGD(model.parameter_groups(), lr=0.1, momentum=0.9)

start = time.time()
num_epochs = 5
for epoch in range(1, num_epochs + 1):

    model.train()

    # Track cumulative loss and accuracy for the training epoch
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, (data, target) in enumerate(train_loader):

        # Convert only data to LNSTensor, target remains a regular tensor
        # since it is an integer tensor for classification.
        data, target = xltorch.lnstensor(data.to(device)), target.to(device)
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

            data, target = xltorch.lnstensor(data.to(device)), target.to(device)

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

        data, target = xltorch.lnstensor(data.to(device)), target.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs.value, dim=1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

print("\nFinal Test Accuracy:", correct / total)
print("Elapsed time:", elapsed)