"""Modified from PyTorch's example https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial"""
import os
import zipfile
import urllib.request
import glob
import unicodedata
import string
import time
import random
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch.utils.data import Dataset
import xlnstorch as xlt

parser = argparse.ArgumentParser(description='MNIST training with LNS')
parser.add_argument('--precision', '-f', type=int, default=None, help='Precision for LNS computations')
parser.add_argument('--base', '-b', type=float, default=None, help='Base for LNS computations')
parser.add_argument('--table', '-t', type=bool, default=True, help='Whether to use table-based LNS computations')
args = parser.parse_args()

if args.table:
    xlt.set_default_sbdb_implementation("tab")
    xlt.operators.implementations.tab.get_table("tmp", f=args.precision, b=args.base)

DATASET_URL = "https://download.pytorch.org/tutorial/data.zip"
ZIP_FILENAME = "data.zip"
EXTRACT_DIR = "."

def download_and_extract():
    if os.path.exists(os.path.join(EXTRACT_DIR, "data", "names")):
        print("Dataset already exists, skipping download.")
        return

    print("Downloading dataset...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_FILENAME)
    print("Extracting contents...")
    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    os.remove(ZIP_FILENAME)
    os.remove(os.path.join(EXTRACT_DIR, "data", "eng-fra.txt"))
    print(f"Dataset ready under '{EXTRACT_DIR}/names/'")

def find_files(path_pattern="data/names/*.txt"):
    return glob.glob(path_pattern)

allowed_characters = string.ascii_letters + " .,;'-"
n_letters = len(allowed_characters)

def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = xlt.zeros(len(line), 1, n_letters, f=args.precision, b=args.base)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir #for provenance of the dataset
        self.load_time = time.localtime #for provenance of the dataset
        labels_set = set() #set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(line_to_tensor(name))
                self.labels.append(label)

        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item

class LNSCharRNN(xlt.nn.LNSModule):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.rnn = xlt.nn.LNSRNN(input_size, hidden_size, num_layers,
                                 weight_f=args.precision, bias_f=args.precision,
                                 weight_b=args.base, bias_b=args.base)
        self.fc = xlt.nn.LNSLinear(hidden_size, output_size,
                                   weight_f=args.precision, bias_f=args.precision,
                                   weight_b=args.base, bias_b=args.base)

        xlt.nn.init.normal_(self.rnn.weight_ih_l0, mean=0.0, std=0.1)
        xlt.nn.init.normal_(self.rnn.weight_hh_l0, mean=0.0, std=0.1)
        xlt.nn.init.zeros_(self.rnn.bias_ih_l0)
        xlt.nn.init.zeros_(self.rnn.bias_hh_l0)

        xlt.nn.init.normal_(self.fc.weight, mean=0.0, std=0.1)
        xlt.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = hidden[-1]

        output = torch.nn.functional.log_softmax(self.fc(hidden), dim=1)
        return output

def label_from_output(output, output_labels):
    top_n, top_i = torch.max(output, dim=1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

device = "cpu"
download_and_extract()

dataset = NamesDataset(os.path.join(EXTRACT_DIR, "data", "names"))
train_split, test_split = 0.85, 0.15
train_set, test_set = torch.utils.data.random_split(
    dataset, [train_split, test_split],
    generator=torch.Generator(device=device).manual_seed(2024)
)
print(f"Dataset loaded with {len(train_set)} training and {len(test_set)} testing examples.")

batch_size = 128
model = LNSCharRNN(n_letters, 64, 1, len(dataset.labels_uniq)).to(device)
criterion = torch.nn.NLLLoss()
optimizer = xlt.optim.LNSAdam(model.lns_parameters(), lr=0.01)

print("Starting training...")
start = time.time()
num_epochs = 10
for epoch in range(1, num_epochs + 1):

    model.train()

    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    batches = list(range(len(train_set)))
    random.shuffle(batches)
    batches = np.array_split(batches, len(batches) // batch_size)

    batch_group_start = time.time()

    for i, batch in enumerate(batches):

        optimizer.zero_grad()

        batch_loss = 0
        batch_correct = 0

        # process each example in the batch individually
        # since they're variable-length sequences
        for example in batch:
            target, data, _, _ = train_set[example]
            output = model(data)
            loss = criterion(output, target)
            batch_loss += loss

            if torch.max(output, dim=1)[1].item() == target:
                batch_correct += 1

        batch_loss.backward()
        optimizer.step()

        running_train_loss += batch_loss.item()
        train_total += len(batch)
        train_correct += batch_correct

        if (i + 1) % 10 == 0:
            batch_group_end = time.time()
            print(f"Batch {i+1}: {batch_correct}/{len(batch)} correct ({(batch_group_end - batch_group_start):.2f}s).")
            batch_group_start = time.time()

    train_epoch_loss = running_train_loss / train_total
    train_epoch_acc = train_correct / train_total

    model.eval()

    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    batches = list(range(len(test_set)))
    random.shuffle(batches)
    batches = np.array_split(batches, len(batches) // batch_size)

    with torch.no_grad():

        for batch in batches:

            batch_loss = 0
            batch_correct = 0

            for example in batch:
                target, data, _, _ = test_set[example]
                output = model(data)
                loss = criterion(output, target)
                batch_loss += loss

                if torch.max(output, dim=1)[1].item() == target:
                    batch_correct += 1

            running_val_loss += batch_loss.item()
            val_total += len(batch)
            val_correct += batch_correct

    val_epoch_loss = running_val_loss / val_total
    val_epoch_acc = val_correct / val_total

    print(f"Epoch {epoch}:")
    print(f"  Training   - Loss = {train_epoch_loss:.4f}, Accuracy = {train_epoch_acc:.4f}")
    print(f"  Validation - Loss = {val_epoch_loss:.4f}, Accuracy = {val_epoch_acc:.4f}")

elapsed = time.time() - start
print(f"Training completed in {elapsed:.2f} seconds.")

print("Plotting confusion matrix...")
confusion = torch.zeros(len(dataset.labels_uniq), len(dataset.labels_uniq))
model.eval()

with torch.no_grad():

    for i in range(len(test_set)):
        target, data, label, text = test_set[i]
        output = model(data)
        guess, guess_i = label_from_output(output, dataset.labels_uniq)
        label_i = dataset.labels_uniq.index(label)
        confusion[label_i][guess_i] += 1

    for i in range(len(dataset.labels_uniq)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.cpu().numpy())
fig.colorbar(cax)

ax.set_xticks(np.arange(len(dataset.labels_uniq)), labels=dataset.labels_uniq, rotation=90)
ax.set_yticks(np.arange(len(dataset.labels_uniq)), labels=dataset.labels_uniq)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()
