import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from argparse import Namespace

from torchinfo import summary

class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    From: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    From: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes
    """
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_V = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_U = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_V.append(nn.Dropout(0.25))
            self.attention_U.append(nn.Dropout(0.25))

        self.attention_V = nn.Sequential(*self.attention_V)
        self.attention_U = nn.Sequential(*self.attention_U)

        self.attention_w = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_V(x)
        b = self.attention_U(x)
        A = a.mul(b)
        A = self.attention_w(A)  # N x n_classes
        return A, x
    
class ADMIL_Model(nn.Module):
    """
    Simple ADMIL model
    Args:
        n_classes: number of output classes, this should be the number of time bins
        hidden_layers: number of hidden layers in classifier, default = 0
        feature_size: size of feature vector (default: 2048)
        dropout: boolean, if True use dropout 
    """
    def __init__(self,
                 n_classes,
                 hidden_layers=0,
                 feature_size=2048,
                 dropout=False):
        super(ADMIL_Model, self).__init__()
        self.dropout = dropout

        fc = [nn.BatchNorm1d(feature_size), nn.Linear(feature_size, 1024), nn.ReLU()]
        
        if self.dropout:
            fc.append(nn.Dropout(0.25))

        # Always use one class for attention, we want only one attention weight per instance
        attn_net = Attn_Net_Gated(L=1024, D=512, dropout=self.dropout, n_classes=1)
        fc.append(attn_net)
        self.gated_attention = nn.Sequential(*fc)

        # Baseline classifier
        if hidden_layers > 0:
            fc2 = []
            size = 1024
            for i in range(hidden_layers):
                fc2.append(nn.Linear(size, int(size/2)))
                fc2.append(nn.ReLU())
                if self.config.dropout:
                    fc2.append(nn.Dropout(0.25))
                size = int(size/2)
            fc2.append(nn.Linear(size, n_classes))
            self.classifier = nn.Sequential(*fc2)
        else:
            self.classifier = nn.Linear(1024, n_classes)

        # can implement a different threshold for classification here
        self.threshold = 0.5

    def forward(self, x):
        x_hat = x.squeeze()
        if x_hat.ndim < 2:
            x_hat = x_hat.unsqueeze(0)
        
        a, x_hat = self.gated_attention(x_hat)  # NxK (batch x num classes)
        a = F.softmax(a, dim=0)  # softmax over N
        h = torch.mm(torch.transpose(a, 1, 0), x_hat)

        logits = self.classifier(h)
        y_prob = torch.sigmoid(logits)
        y_hat = torch.where(y_prob > self.threshold, 1, 0)

        return logits, y_prob, y_hat
    

def test_admil():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = ADMIL_Model(
        n_classes = 10,
        hidden_layers=0,
        feature_size=1024,
        dropout=False
    )

    x = torch.randn(1, 139, 1024).to(device)
    logits, probs, hats = model(x)

    summary(model, None)

if __name__ == "__main__":
    test_admil()
