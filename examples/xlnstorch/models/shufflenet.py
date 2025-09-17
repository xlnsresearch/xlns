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


class ShuffleV2Block(nn.LNSModule):

    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.stride = stride
        branch_out = out_c // 2

        if stride == 1:
            self.branch2 = nn.LNSSequential(
                # 1 × 1 pw-conv
                nn.LNSConv2d(in_c // 2, branch_out, 1, 1, 0, bias=False),
                nn.LNSBatchNorm2d(branch_out),
                torch.nn.ReLU(),

                # 3 × 3 dw-conv
                nn.LNSConv2d(branch_out, branch_out, 3, 1, 1,
                             groups=branch_out, bias=False),
                nn.LNSBatchNorm2d(branch_out),

                # 1 × 1 pw-conv
                nn.LNSConv2d(branch_out, branch_out, 1, 1, 0, bias=False),
                nn.LNSBatchNorm2d(branch_out),
                torch.nn.ReLU(),
            )

        else:
            self.branch1 = nn.LNSSequential(
                # 3 × 3 dw-conv
                nn.LNSConv2d(in_c, in_c, 3, stride, 1,
                             groups=in_c, bias=False),
                nn.LNSBatchNorm2d(in_c),

                # 1 × 1 pw-conv
                nn.LNSConv2d(in_c, branch_out, 1, 1, 0, bias=False),
                nn.LNSBatchNorm2d(branch_out),
                torch.nn.ReLU(),
            )

            self.branch2 = nn.LNSSequential(
                # 1 × 1 pw-conv
                nn.LNSConv2d(in_c, branch_out, 1, 1, 0, bias=False),
                nn.LNSBatchNorm2d(branch_out),
                torch.nn.ReLU(),

                # 3 × 3 dw-conv
                nn.LNSConv2d(branch_out, branch_out, 3, stride, 1,
                             groups=branch_out, bias=False),
                nn.LNSBatchNorm2d(branch_out),

                # 1 × 1 pw-conv
                nn.LNSConv2d(branch_out, branch_out, 1, 1, 0, bias=False),
                nn.LNSBatchNorm2d(branch_out),
                torch.nn.ReLU(),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), 1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)

        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.LNSModule):

    stage_repeats = [4, 8, 4]

    def __init__(self, num_classes = 10, in_channels = 3, dropout = 0.0):
        super().__init__()

        out_channels = [24, 116, 232, 464, 1024]

        self.conv1 = nn.LNSSequential(
            nn.LNSConv2d(in_channels, out_channels[0], 3, 2, 1, bias=False),
            nn.LNSBatchNorm2d(out_channels[0]),
            torch.nn.ReLU(inplace=True)
        )
        self.maxpool = nn.LNSMaxPool2d(kernel_size=3, stride=2, padding=1)

        input_c = out_channels[0]
        stage_idx = 0
        blocks = []
        for repeats, output_c in zip(self.stage_repeats, out_channels[1:-1]):
            blocks.append(ShuffleV2Block(input_c, output_c, stride=2))
            input_c = output_c
            for _ in range(repeats - 1):
                blocks.append(ShuffleV2Block(input_c, output_c, stride=1))
            stage_idx += 1
        self.stages = nn.LNSSequential(*blocks)

        self.conv5 = nn.LNSSequential(
            nn.LNSConv2d(input_c, out_channels[-1], 1, 1, 0, bias=False),
            nn.LNSBatchNorm2d(out_channels[-1]),
            torch.nn.ReLU(inplace=True)
        )

        self.pool = nn.LNSAdaptiveAvgPool2d(1)
        self.dropout = nn.LNSDropout(dropout)
        self.classifier = nn.LNSLinear(out_channels[-1], num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
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

if __name__ == "__main__":

    # note: H x W should be sufficiently large so that the feature map is
    # not too small after the downsampling operations in the network

    model_v1 = ShuffleNetV1()
    inp = xlt.randn(2, 3, 48, 48) # (batch_size, channels = 3, height, width)

    out = model_v1(inp)
    print(out)


    model_v2 = ShuffleNetV2()
    inp = xlt.randn(2, 3, 48, 48) # (batch_size, channels = 3, height, width)

    out = model_v2(inp)
    print(out)