import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImEDL(nn.Module):
    """
    模型框架
    """
    def __init__(self, channels, num_classes, mem_h):
        super(ImEDL, self).__init__()
        torch.manual_seed(925)
        torch.cuda.manual_seed(925)

        self.net = CnnNet(channels=channels, num_classes=num_classes)
        # self.net = ResNet50(channels=channels, num_classes=num_classes)

        self.w1 = nn.Parameter(torch.rand(1, mem_h))
        self.w2 = nn.Parameter(torch.rand(mem_h, mem_h))
        self.w3 = nn.Parameter(torch.rand(mem_h, 1))

    def forward(self, x):
        # 获取初始证据
        logits = self.net(x)
        evidence_src = nn.Softplus()(logits)
        evidence_src = evidence_src.unsqueeze(-1)

        # 记忆模块
        evidence_input = evidence_src
        w = torch.matmul(evidence_input, self.w1)
        w = nn.ReLU()(w)
        w = torch.matmul(w, self.w2)
        w = nn.ReLU()(w)
        w = torch.matmul(w, self.w3)
        w = nn.Softplus()(w)

        # 计算校准后证据
        evidence = evidence_src * w
        evidence = evidence.squeeze()

        # Dirichlet 分布参数 alpha
        alpha = 1 + evidence

        # 返回结果
        evidence_src = evidence_src.squeeze()
        w = torch.squeeze(w)

        return evidence_src, alpha, w


class CnnNet(nn.Module):
    """
        : CNN
        : input (batch_size, channels, 28, 28)
    """

    def __init__(self, channels, num_classes):
        super(CnnNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x


class ResNet50(nn.Module):
    """
        : ResNet50
        : input (batch_size, 3, 224, 224)
    """

    def __init__(self, channels, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc2(x)

        return x
