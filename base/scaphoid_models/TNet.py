import torch
import torch.nn as nn
import torch.nn.functional as F

from base.scaphoid_utils.logger import print_log


class InputTransformNet(nn.Module):

    def __init__(self, K=3):
        super(InputTransformNet, self).__init__()

        self.K = K
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,1), stride=(1,1), padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(1,1), padding=0)
        self.conv3 = nn.Conv2d(128, 1024, kernel_size=(1,1), stride=(1,1), padding=0)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, K * K)

        self.init_bias = torch.eye(K).view(-1) # init transformation matrix as flattened identity matrix

    def forward(self, point_cloud):
        """
        :param point_cloud: Input point cloud data  of shape [B, 3, N]
        :return: Transformed point cloud data
        """
        B, _, N = point_cloud.shape
        input_image = point_cloud.unsqueeze(1)  # shape [B, 1, 3, N]

        x = F.relu(self.bn1(self.conv1(input_image)))   # shape [B, 64, 1, N]
        x = F.relu(self.bn2(self.conv2(x)))             # shape [B, 128, 1, N]
        x = F.relu(self.bn3(self.conv3(x)))             # shape [B, 1024, 1, N]

        x = F.max_pool2d(x, kernel_size=(1, N))         # shape [B, 1024, 1, 1]

        x = x.view(B, -1)                               # shape [B, 1024]

        x = F.relu(self.bn_fc1(self.fc1(x)))            # shape [B, 512]
        x = F.relu(self.bn_fc2(self.fc2(x)))            # shape [B, 256]

        x = self.fc3(x)                                 # shape [B, K*K]

        x = x + self.init_bias.to(x.device)  # add bias to the output
        transform = x.view(B, self.K, self.K)  # reshape to [B, K, K]

        return transform

    
    
class FeatureTransformNet(nn.Module):

    def __init__(self, input_dim=64, K=64):
        super(FeatureTransformNet, self).__init__()

        self.K = K
        self.conv1 = nn.Conv2d(1, input_dim, kernel_size=(1,1), stride=(1,1), padding=0)
        self.conv2 = nn.Conv2d(input_dim, input_dim*2, kernel_size=(1,1), stride=(1,1), padding=0)
        self.conv3 = nn.Conv2d(input_dim*2, 16*input_dim, kernel_size=(1,1), stride=(1,1), padding=0)

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim*2)
        self.bn3 = nn.BatchNorm2d(16*input_dim)

        self.fc1 = nn.Linear(16*input_dim, 8*input_dim)
        self.fc2 = nn.Linear(8*input_dim, 4*input_dim)

        self.bn_fc1 = nn.BatchNorm1d(8*input_dim)
        self.bn_fc2 = nn.BatchNorm1d(4*input_dim)

        self.fc3 = nn.Linear(4*input_dim, K * K)

        self.init_bias = torch.eye(K).view(-1)


    def forward(self, features):
        """
        :param features: Input feature data of shape [B, 64, N]
        :return: Transformation matrix of shape [B, K, K]
        """

        B, _, N = features.shape
        input_image = features.unsqueeze(2)  # shape [B, 64, 1, N]

        x = F.relu(self.bn1(self.conv1(input_image)))   # shape [B, 64, 1, N]
        x = F.relu(self.bn2(self.conv2(x)))             # shape [B, 128, 1, N]
        x = F.relu(self.bn3(self.conv3(x)))             # shape [B, 1024, 1, N]

        x = F.max_pool2d(x, kernel_size=(1, N))         # shape [B, 1024, 1, 1]
        x = x.view(B, -1)                               # shape [B, 1024]

        x = F.relu(self.bn_fc1(self.fc1(x)))            # shape [B, 512]
        x = F.relu(self.bn_fc2(self.fc2(x)))            # shape [B, 256]

        x = self.fc3(x)                                 # shape [B, K*K]

        x = x + self.init_bias.to(x.device)  # add bias to the output

        transform = x.view(B, self.K, self.K)  # reshape to [B, K, K]

        return transform