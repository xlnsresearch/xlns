import argparse
import torch
import xlnstorch as xltorch
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

    def __init__(self, in_channels=1, out_channels=10, out1=16, out2=32, out_mp2=7*7):
        super().__init__()
        self.conv1 = xltorch.nn.LNSConv2d(in_channels, out1, kernel_size=5, padding=2)
        self.mp1 = xltorch.nn.LNSMaxPool2d(kernel_size=2, stride=2)
        self.conv2 = xltorch.nn.LNSConv2d(out1, out2, kernel_size=5, padding=2)
        self.mp2 = xltorch.nn.LNSMaxPool2d(kernel_size=2, stride=2)
        self.fc = xltorch.nn.LNSLinear(out2 * out_mp2, out_channels)
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

if __name__ == "__main__":

    model = LNSTinyCNN(in_channels=3, out_channels=10)
    inp = xltorch.randn(5, 3, 28, 28) # (batch_size, channels = 3, height, width)

    out = model(inp)
    print(out)