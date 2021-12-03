2import importlib

import torch
import torch.nn as nn


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class ResNet(nn.Module):
    def __init__(self, num_punches=4, num_stances = 2, num_sides=2,  num_channels=3,
                 encoder='resnet34',  last_stance='sigmoid', last_punch='sigmoid', last_side='sigmoid',
                 use_pretrain=False, output_rel=False):
        super().__init__()
        assert num_channels > 0, "Incorrect num channels"
        assert encoder in ['resnet18', 'resnet34', 'resnet50',
                           'resnet101', 'resnet152'],\
            "Incorrect encoder type"
        self.deploy = False  # Set to True on inference with external preproc
        self.output_rel = output_rel  # 3d output - relative punch offset
        resnet = class_for_name("torchvision.models", encoder)(
            pretrained=use_pretrain, progress=False)
        if num_channels != 3:  # Number of input channels
            conv1 = nn.Conv2d(num_channels, 64,
                              kernel_size=(7, 7), stride=(2, 2),
                              padding=(3, 3), bias=False)
        else:
            conv1 = resnet.conv1
        self.firstconv = nn.Sequential(
            conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        if encoder in ['resnet18', 'resnet34']:
            block_expansion = 1
        else:
            block_expansion = 4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_expansion, 512)
        self.fc1 = nn.Linear(512, 128)
        self.punches = nn.Linear(128, num_punches)
        self.fc2 = nn.Linear(512, 128)
        self.sides = nn.Linear(128, num_sides)
        self.relu = nn.ReLU(inplace=True)
        self.last_punch = self._last(last_punch)
        self.last_side = self._last(last_side)

        self.fc3 = nn.Linear(512, 128)
        self.rel = nn.Linear(128, 1)
        self.last_rel = nn.Sigmoid()

        self.fc4 = nn.Linear(512, 128)
        self.stances = nn.Linear(128, num_stances)
        self.last_stance = self._last(last_stance)




    def _last(self, layer_type: str):
        assert layer_type in ['softmax', 'sigmoid'], 'Wrong last layer type'
        if layer_type == 'softmax':
            last = nn.Softmax(dim=1)
        else:
            last = nn.Sigmoid()
        return last

    def forward(self, x):
        # Comment this if block for tflite export
        if not self.deploy:
            x = x.float().div(255.0)
        x = self.firstconv(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.avgpool(x)
        # Please comment this line and uncomment the next for tflite export
        x = torch.flatten(x, 1)
        # x = x.view(1, 512)
        x = self.fc(x)

        x_punches = self.relu(self.fc1(x))
        x_punches = self.punches(x_punches)
        x_punches = self.last_punch(x_punches)

        x_sides = self.relu(self.fc2(x))
        x_sides = self.sides(x_sides)
        x_sides = self.last_side(x_sides)
        x_rel = self.relu(self.fc3(x))
        x_rel = self.rel(x_rel)
        x_rel = self.last_rel(x_rel)

        x_stances = self.relu(self.fc4(x))
        x_stances = self.sides(  x_stances)
        x_stances = self.last_side(  x_stances)
        return x_punches, x_sides, x_rel, x_stances
