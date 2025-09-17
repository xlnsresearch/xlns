import xlnstorch as xlt
from xlnstorch import nn
import torch
import xlns
import argparse

# Example precision and base settings (these are mutually exclusive)
parser = argparse.ArgumentParser(description='ResNet18 model with LNS')
parser.add_argument('--precision', '-f', type=int, default=None, help='Precision for LNS computations')
parser.add_argument('--base', '-b', type=float, default=None, help='Base for LNS computations')
parser.add_argument('--table', '-t', type=bool, default=False, help='Whether to use table-based LNS computations')
args = parser.parse_args()

if args.table:
    if args.precision is None and args.base is None:
        raise ValueError("Must specify precision or base with --table option")
    xlt.set_default_sbdb_implementation("tab")
    xlt.operators.tab.get_table("tmp", f=args.precision, b=args.base)

elif args.precision is None and args.base is not None:
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

if __name__ == "__main__":

    model = MobileNetV1()
    inp = xlt.randn(2, 3, 24, 24) # (batch_size, channels = 3, height, width)

    out = model(inp)
    print(out)