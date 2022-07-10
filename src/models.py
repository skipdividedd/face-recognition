import torch.nn as nn
import torch
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
import math

device = 'cpu'

def model(md_name):
    if md_name == 'resnet':
        resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=500).to(device)
        # freezing first 11 layers
        ct = 0
        for child in resnet.children():
            ct += 1
            if ct < 11:
                for param in child.parameters():
                    param.requires_grad = False

        return resnet

    elif md_name == 'Triplet':
        model_triplet = InceptionResnetV1(pretrained='vggface2',
                                          classify=False).to(device)
        ct = 0
        for child in model_triplet.children():
            ct += 1
            if ct < 11:
                for param in child.parameters():
                    param.requires_grad = False

        return model_triplet

    elif md_name == 'ArcFace':
        class ArcMarginProduct(nn.Module):
            r"""Implement of large margin arc distance: :
                Args:
                    in_features: size of each input sample
                    out_features: size of each output sample
                    s: norm of input feature
                    m: margin
                    cos(theta + m)
                """

            def __init__(self, in_features, out_features, s=30.0, m=2, easy_margin=False, ls_eps=0.0):
                super(ArcMarginProduct, self).__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.s = s
                self.m = m
                self.ls_eps = ls_eps  # label smoothing
                self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
                nn.init.xavier_uniform_(self.weight)

                self.easy_margin = easy_margin
                self.cos_m = math.cos(m)
                self.sin_m = math.sin(m)
                self.th = math.cos(math.pi - m)
                self.mm = math.sin(math.pi - m) * m

            def forward(self, input, label):
                # --------------------------- cos(theta) & phi(theta) ---------------------------
                cosine = F.linear(F.normalize(input), F.normalize(self.weight))
                sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
                phi = cosine * self.cos_m - sine * self.sin_m
                if self.easy_margin:
                    phi = torch.where(cosine > 0, phi, cosine)
                else:
                    phi = torch.where(cosine > self.th, phi, cosine - self.mm)
                # --------------------------- convert label to one-hot ---------------------------
                # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
                one_hot = torch.zeros(cosine.size(), device=device)
                one_hot.scatter_(1, label.view(-1, 1).long(), 1)
                if self.ls_eps > 0:
                    one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
                # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
                output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
                output *= self.s

                return output

        resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=500).to(device)
        ct = 0
        for child in resnet.children():
            ct += 1
            if ct < 11:
                for param in child.parameters():
                    param.requires_grad = False
        resnet.classify = ArcMarginProduct(512, 500)
        return resnet
