from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn

class EcgCNN(nn.Module):
    def __init__(self):
        super(EcgCNN, self).__init__()

        # First convolutional block
        self.conv_1 = nn.Conv1d(1, 32, 32)
        self.conv_2 = nn.Conv1d(32, 32, 32)
        self.residual_conv1 = nn.Conv1d(1, 32, kernel_size=1)

        # Second convolutional block
        self.conv_3 = nn.Conv1d(32, 64, 16)
        self.conv_4 = nn.Conv1d(64, 64, 16)
        self.residual_conv2 = nn.Conv1d(32, 64, kernel_size=1)

        # Third convolutional block
        self.conv_5 = nn.Conv1d(64, 128, 8)
        self.conv_6 = nn.Conv1d(128, 128, 8)
        self.residual_conv3 = nn.Conv1d(64, 128, kernel_size=1)

        # Pooling layers
        self.pool_1 = nn.MaxPool1d(8, 2)
        self.pool_2 = nn.MaxPool1d(8, 2)
        self.pool_3 = nn.MaxPool1d(635)  # Adjust if necessary

        # Activation and dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        # First residual block
        residual = x  # Shape: [batch_size, 1, 2560]

        # Apply asymmetric padding before convolutions
        x = F.pad(x, (16, 15))
        x = self.conv_1(x)  # Output shape will match residual after padding
        x = self.relu(x)
        x = self.dropout(x)

        x = F.pad(x, (16, 15))
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        if residual.shape[1] != x.shape[1]:
            residual = self.residual_conv1(residual)
        x = x + residual  # Shapes should match

        x = self.pool_1(x)

        # Second residual block
        residual = x

        x = F.pad(x, (8, 7))
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = F.pad(x, (8, 7))
        x = self.conv_4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        if residual.shape[1] != x.shape[1]:
            residual = self.residual_conv2(residual)
        x = x + residual

        x = self.pool_2(x)

        # Third residual block
        residual = x

        x = F.pad(x, (4, 3))
        x = self.conv_5(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = F.pad(x, (4, 3))
        x = self.conv_6(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        if residual.shape[1] != x.shape[1]:
            residual = self.residual_conv3(residual)
        x = x + residual

        x = self.pool_3(x)
        x = x.squeeze(dim=2)

        return x

class EcgHead(nn.Module):
    def __init__(self, n_out=1, drop_rate=0.6):
        super(EcgHead, self).__init__()

        self.head_1 = nn.Linear(128, 128)
        self.head_3 = nn.Linear(128, n_out)
        self.dropout = nn.Dropout(drop_rate)
        self.out_activation = None  # Use activation in loss function if needed

    def forward(self, x):
        x = self.head_1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.head_3(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

class EcgAmigosHead(nn.Module):
    def __init__(self, n_out=1, drop_rate=0.4):
        super(EcgAmigosHead, self).__init__()

        self.head_1 = nn.Linear(128, 64)
        self.head_4 = nn.Linear(64, n_out)
        self.dropout = nn.Dropout(drop_rate)
        self.out_activation = None  # Use activation in loss function if needed

    def forward(self, x):
        x = self.head_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.head_4(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

class EcgNetwork(nn.Module):
    def __init__(self, n_task_heads, n_emotions):
        super(EcgNetwork, self).__init__()

        self.cnn = EcgCNN()

        self.task_heads = nn.ModuleList([EcgHead() for _ in range(n_task_heads)])
        self.emotion_head = EcgHead(n_out=n_emotions)

        self.is_pretext = True

    def forward(self, x):
        embedding = self.cnn(x)

        if self.is_pretext:
            x_list = [th(embedding) for th in self.task_heads]
            out_stacked = torch.stack(x_list)
            return out_stacked, embedding
        else:
            x = self.emotion_head(embedding)
            return x, embedding

class AveragePretextLoss(nn.Module):
    def __init__(self, per_task_criterion, coefficients):
        super(AveragePretextLoss, self).__init__()
        self.per_task_criterion = per_task_criterion
        self.coefficients = coefficients

    def forward(self, outputs, labels):
        total_loss = 0
        for i in range(len(outputs)):
            loss = self.per_task_criterion(outputs[i], labels[:, i]) * self.coefficients[i]
            total_loss += loss
        return total_loss / len(outputs)

def labels_to_vec(labels, n_tasks, debug_ltv=False):
    binary_matrix = torch.zeros((len(labels), n_tasks))
    for i in range(n_tasks):
        l_vec = (labels == i).int().float()
        binary_matrix[:, i] = l_vec
    if debug_ltv:
        print(binary_matrix)
        print(binary_matrix.shape)
    return binary_matrix

