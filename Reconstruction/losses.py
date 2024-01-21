import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class GetSobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        kernel_v = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)
        return x


class GetLaplace(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i = F.conv2d(x_i.unsqueeze(1), self.weight, padding=1)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)
        return x


class GetHighPass(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]
        kernel_h = [[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class Getgradientnopadding(nn.Module):
    def __init__(self):
        super(Getgradientnopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


def ln_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    loss_2 = nn.MSELoss()
    return loss_1(prediction_results, ground_truth) + \
           5 * loss_2(prediction_results, ground_truth)


def sharp_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()

    get_grad = Getgradientnopadding().cuda()
    get_sobel = GetSobel().cuda()
    get_laplace = GetLaplace().cuda()
    get_high = GetHighPass().cuda()

    loss = 0
    loss += loss_1(get_grad(prediction_results), get_grad(ground_truth))
    loss += loss_1(get_sobel(prediction_results), get_sobel(ground_truth))
    loss += loss_1(get_laplace(prediction_results), get_laplace(ground_truth))
    loss += loss_1(get_high(prediction_results), get_high(ground_truth))
    return loss / 4


def grad_loss_simple(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    grad_pre = prediction_results[:, :, 1:, 1:, 1:] - prediction_results[:, :, :-1, :-1, :-1]
    grad_gt = ground_truth[:, :, 1:, 1:, 1:] - ground_truth[:, :, :-1, :-1, :-1]
    return loss_1(grad_pre, grad_gt)


def reconstruction_loss(prediction_results, ground_truth, lamb=0.5, dim="2d"):
    if dim == "2d":
        return ln_loss(prediction_results, ground_truth) + \
           lamb * sharp_loss(prediction_results, ground_truth)
    else:
        # print(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1).shape,
        #       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1).shape,)
        if lamb == 0:
            return ln_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1),
                           ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1))
        else:
            return ln_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1),
                           ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1)) + \
                   lamb * sharp_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1),
                                     ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1))


def flow_loss(prediction_results, ground_truth, dim="2d"):
    if dim == "2d":
        return ln_loss(prediction_results[:, 1:] - prediction_results[:, :-1],
                       ground_truth[:, 1:] - ground_truth[:, :-1])
    else:
        return ln_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1)[:, 1:] -
                       prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1)[:, :-1],
                       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1)[:, 1:] -
                       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1)[:, :-1])



class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss