import xlnstorch as xlt
from xlnstorch import nn
import torch
import xlns

# Example precision and base settings (these are mutually exclusive)
f = 23
b = None

if f is None and b is None:
    raise ValueError("Either precision (f) or base (b) must be specified.")
elif f is None:
    xlns.xlnsB = b
elif b is None:
    xlns.xlnssetF(f)

class BasicBlock(nn.LNSModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.LNSConv2d(in_channels, out_channels, kernel_size=3,
                                  stride=stride, padding=1, bias=False)
        self.bn1 = nn.LNSBatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = nn.LNSConv2d(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
        self.bn2 = nn.LNSBatchNorm2d(out_channels)

        self.shortcut = nn.LNSSequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.LNSSequential(
                nn.LNSConv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                nn.LNSBatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.LNSModule):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.LNSConv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        self.bn1 = nn.LNSBatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = nn.LNSMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.LNSAdaptiveAvgPool2d((1, 1))
        self.fc = nn.LNSLinear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.LNSSequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return torch.nn.functional.log_softmax(out, dim=1)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.LNSConv2d):
                xlt.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    xlt.nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LNSLinear):
                xlt.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    xlt.nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LNSBatchNorm2d):
                xlt.nn.init.ones_(m.weight)
                xlt.nn.init.zeros_(m.bias)

        # zero-initialize the last BN in each residual block's residual branch
        for m in self.modules():
            if isinstance(m, BasicBlock):
                xlt.nn.init.zeros_(m.bn2.weight)

if __name__ == "__main__":

    model = ResNet18()
    inp = xlt.randn(2, 3, 24, 24) # (batch_size, channels = 3, height, width)

    out = model(inp)
    print(out)