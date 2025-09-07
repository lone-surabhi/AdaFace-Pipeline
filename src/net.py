from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout, MaxPool2d, Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid, Module, PReLU


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))

class GNAP(Module):
    def __init__(self, in_c):
        super().__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)
    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        weight = x_norm.mean() / (x_norm + 1e-6)
        x = x * weight
        x = self.pool(x).view(x.shape[0], -1)
        return self.bn2(x)

class GDC(Module):
    def __init__(self, in_c, embedding_size):
        super().__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c, groups=in_c, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)
    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.flatten(x)
        x = self.linear(x)
        return self.bn(x)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = Sigmoid()
    def forward(self, x):
        s = self.avg_pool(x)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s

class BasicBlockIR(Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        if in_ch == out_ch:
            self.shortcut = MaxPool2d(1, stride)
        else:
            self.shortcut = Sequential(Conv2d(in_ch, out_ch, 1, stride, bias=False), BatchNorm2d(out_ch))
        self.res = Sequential(
            BatchNorm2d(in_ch),
            Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            BatchNorm2d(out_ch),
            PReLU(out_ch),
            Conv2d(out_ch, out_ch, 3, stride, 1, bias=False),
            BatchNorm2d(out_ch),
        )
    def forward(self, x):
        return self.res(x) + self.shortcut(x)

class BottleneckIR(Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        mid = out_ch // 4
        if in_ch == out_ch:
            self.shortcut = MaxPool2d(1, stride)
        else:
            self.shortcut = Sequential(Conv2d(in_ch, out_ch, 1, stride, bias=False), BatchNorm2d(out_ch))
        self.res = Sequential(
            BatchNorm2d(in_ch),
            Conv2d(in_ch, mid, 1, 1, 0, bias=False),
            BatchNorm2d(mid),
            PReLU(mid),
            Conv2d(mid, mid, 3, 1, 1, bias=False),
            BatchNorm2d(mid),
            PReLU(mid),
            Conv2d(mid, out_ch, 1, stride, 0, bias=False),
            BatchNorm2d(out_ch),
        )
    def forward(self, x):
        return self.res(x) + self.shortcut(x)

class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__(in_ch, out_ch, stride)
        self.res.add_module("se", SEModule(out_ch, 16))

class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__(in_ch, out_ch, stride)
        self.res.add_module("se", SEModule(out_ch, 16))

BottleneckCfg = namedtuple('Block', ['in_channel', 'depth', 'stride'])

def _make_block(in_ch, out_ch, num_units, stride=2):
    return [BottleneckCfg(in_ch, out_ch, stride)] + [BottleneckCfg(out_ch, out_ch, 1) for _ in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 18:
        return [
            _make_block(64, 64, 2),
            _make_block(64, 128, 2),
            _make_block(128, 256, 2),
            _make_block(256, 512, 2),
        ]
    if num_layers == 34:
        return [
            _make_block(64, 64, 3),
            _make_block(64, 128, 4),
            _make_block(128, 256, 6),
            _make_block(256, 512, 3),
        ]
    if num_layers == 50:
        return [
            _make_block(64, 64, 3),
            _make_block(64, 128, 4),
            _make_block(128, 256, 14),
            _make_block(256, 512, 3),
        ]
    if num_layers == 100:  
        return [
            _make_block(64, 64, 3),
            _make_block(64, 128, 13),
            _make_block(128, 256, 30),
            _make_block(256, 512, 3),
        ]
    if num_layers == 152:
        return [
            _make_block(64, 256, 3),
            _make_block(256, 512, 8),
            _make_block(512, 1024, 36),
            _make_block(1024, 2048, 3),
        ]
    if num_layers == 200:
        return [
            _make_block(64, 256, 3),
            _make_block(256, 512, 24),
            _make_block(512, 1024, 36),
            _make_block(1024, 2048, 3),
        ]
    raise ValueError("Unsupported num_layers")

class Backbone(Module):
    def __init__(self, input_size=(112,112), num_layers=50, mode='ir'):
        super().__init__()
        assert input_size[0] in [112, 224] and input_size[1] in [112, 224]
        assert num_layers in [18, 34, 50, 100, 152, 200]
        assert mode in ['ir', 'ir_se']

        self.input_layer = Sequential(Conv2d(3, 64, 3, 1, 1, bias=False), BatchNorm2d(64), PReLU(64))

        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            unit = BasicBlockIR if mode == 'ir' else BasicBlockIRSE
            out_ch = 512
        else:
            unit = BottleneckIR if mode == 'ir' else BottleneckIRSE
            out_ch = 2048

        modules = []
        for stage in blocks:
            for cfg in stage:
                modules.append(unit(cfg.in_channel, cfg.depth, cfg.stride))
        self.body = Sequential(*modules)

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(out_ch),
                Dropout(0.4),
                Flatten(),
                Linear(out_ch * 7 * 7, 512, bias=False),
                BatchNorm1d(512, affine=False)
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(out_ch),
                Dropout(0.4),
                Flatten(),
                Linear(out_ch * 14 * 14, 512, bias=False),
                BatchNorm1d(512, affine=False)
            )

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        out = x / (norm + 1e-6)  
        return out, norm

def IR_18(input_size=(112,112)):  return Backbone(input_size, 18, 'ir')
def IR_34(input_size=(112,112)):  return Backbone(input_size, 34, 'ir')
def IR_50(input_size=(112,112)):  return Backbone(input_size, 50, 'ir')
def IR_101(input_size=(112,112)): return Backbone(input_size, 100, 'ir')
def IR_152(input_size=(112,112)): return Backbone(input_size, 152, 'ir')
def IR_200(input_size=(112,112)): return Backbone(input_size, 200, 'ir')

def IR_SE_50(input_size=(112,112)):  return Backbone(input_size, 50, 'ir_se')
def IR_SE_101(input_size=(112,112)): return Backbone(input_size, 100, 'ir_se')
def IR_SE_152(input_size=(112,112)): return Backbone(input_size, 152, 'ir_se')
def IR_SE_200(input_size=(112,112)): return Backbone(input_size, 200, 'ir_se')

def build_model(model_name='ir_50'):
    name = model_name.lower()
    if name == 'ir_18': return IR_18()
    if name == 'ir_34': return IR_34()
    if name == 'ir_50': return IR_50()
    if name == 'ir_101': return IR_101()
    if name == 'ir_152': return IR_152()
    if name == 'ir_200': return IR_200()
    if name == 'ir_se_50': return IR_SE_50()
    if name == 'ir_se_101': return IR_SE_101()
    if name == 'ir_se_152': return IR_SE_152()
    if name == 'ir_se_200': return IR_SE_200()
    raise ValueError(f"Unknown model name: {model_name}")