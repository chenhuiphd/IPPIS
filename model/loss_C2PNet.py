import torch.nn.functional as F
import torch.nn as nn
import torch
from math import exp
# def l1_and_l2(P_A_pred, P_A_gt, P_T_pred, P_T_gt, T_pred, T_gt, **kwargs):
#     P_A_l1_loss_lambda = kwargs.get('P_A_l1_loss_lambda', 1)
#     P_A_l1_loss = F.l1_loss(P_A_pred, P_A_gt) * P_A_l1_loss_lambda
#
#     P_A_l2_loss_lambda = kwargs.get('P_A_l2_loss_lambda', 1)
#     P_A_l2_loss = F.mse_loss(P_A_pred, P_A_gt) * P_A_l2_loss_lambda
#
#     P_T_l1_loss_lambda = kwargs.get('P_T_l1_loss_lambda', 1)
#     P_T_l1_loss = F.l1_loss(P_T_pred, P_T_gt) * P_T_l1_loss_lambda
#
#     P_T_l2_loss_lambda = kwargs.get('P_T_l2_loss_lambda', 1)
#     P_T_l2_loss = F.mse_loss(P_T_pred, P_T_gt) * P_T_l2_loss_lambda
#
#     T_l1_loss_lambda = kwargs.get('T_l1_loss_lambda', 1)
#     T_l1_loss = F.l1_loss(T_pred, T_gt) * T_l1_loss_lambda
#
#     T_l2_loss_lambda = kwargs.get('T_l2_loss_lambda', 1)
#     T_l2_loss = F.mse_loss(T_pred, T_gt) * T_l2_loss_lambda
#
#     print('P_A_l1_loss:', P_A_l1_loss.item())
#     print('P_A_l2_loss:', P_A_l2_loss.item())
#     print('P_T_l1_loss:', P_T_l1_loss.item())
#     print('P_T_l2_loss:', P_T_l2_loss.item())
#     print('T_l1_loss:', T_l1_loss.item())
#     print('T_l2_loss:', T_l2_loss.item())
#
#     return P_A_l1_loss + P_A_l2_loss + P_T_l1_loss + P_T_l2_loss + T_l1_loss + T_l2_loss

def l1_and_l2(R_pred, R_gt, **kwargs):
    R_l2_loss_lambda = kwargs.get('R_l2_loss_lambda', 1)
    R_l2_loss = F.mse_loss(R_pred, R_gt) +1-SSIM()(R_pred, R_gt)
    #print('R_l2_loss:', R_l2_loss.item())
    return R_l2_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class TVLoss(nn.Module):
    """
        total variation (TV) loss encourages spatial smoothness
    """

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        N, C, H, W = x.size()
        count_h = C * (H - 1) * W
        count_w = C * H * (W - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :H - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :W - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / N