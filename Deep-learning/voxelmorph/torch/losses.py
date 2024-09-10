import torch
import torch.nn.functional as F
import numpy as np
import math
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv3d

EPSILON = 1e-8


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Drloc():


    def relative_constraint_l1(self, deltaxy, predxy):
        return F.l1_loss(deltaxy, predxy)

    def relative_constraint_ce(self, deltaxy, predxy):
        # predx, predy = torch.chunk(predxy, chunks=2, dim=1)
        predx, predy = predxy[:, :, 0], predxy[:, :, 1]
        targetx, targety = deltaxy[:, 0].long(), deltaxy[:, 1].long()
        return F.cross_entropy(predx, targetx) + F.cross_entropy(predy, targety)

    def variance_aware_regression(self, pred, beta, target, labels, lambda_var=0.001):
        # Variance aware regression.
        pred_titled = pred.unsqueeze(0).t().repeat(1, labels.size(1))
        pred_var = torch.sum((labels - pred_titled) ** 2 * beta, dim=1) + EPSILON
        pred_log_var = torch.log(pred_var)
        squared_error = (pred - target) ** 2
        return torch.mean(torch.exp(-pred_log_var) * squared_error + lambda_var * pred_log_var)

    # based on the codes: https://github.com/google-research/google-research/blob/master/tcc/tcc/losses.py
    def relative_constraint_cbr(self, deltaxy, predxy, loss_type="regression_mse_var"):
        predx, predy = predxy[:, :, 0], predxy[:, :, 1]
        num_classes = predx.size(1)
        targetx, targety = deltaxy[:, 0].long(), deltaxy[:, 1].long()  # [N, ], [N, ]
        betax, betay = F.softmax(predx, dim=1), F.softmax(predy, dim=1)  # [N, C], [N, C]
        labels = torch.arange(num_classes).unsqueeze(0).to(predxy.device)  # [1, C]
        true_idx = targetx  # torch.sum(targetx*labels, dim=1)      # [N, ]
        true_idy = targety  # torch.sum(targety*labels, dim=1)      # [N, ]

        pred_idx = torch.sum(betax * labels, dim=1)  # [N, ]
        pred_idy = torch.sum(betay * labels, dim=1)  # [N, ]

        if loss_type in ["regression_mse", "regression_mse_var"]:
            if "var" in loss_type:
                # Variance aware regression.
                lossx = Drloc.variance_aware_regression(pred_idx, betax, true_idx, labels)
                lossy = Drloc.variance_aware_regression(pred_idy, betay, true_idy, labels)
            else:
                lossx = torch.mean((pred_idx - true_idx) ** 2)
                lossy = torch.mean((pred_idy - true_idy) ** 2)
            loss = lossx + lossy
            return loss
        else:
            raise NotImplementedError("We only support regression_mse and regression_mse_var now.")

    def cal_selfsupervised_loss(self, outs, args, lambda_drloc=0.0):
        loss, all_losses = 0.0, Munch()
        if args.TRAIN.USE_DRLOC:
            if args.TRAIN.DRLOC_MODE == "l1":  # l1 regression constraint
                reld_criterion = Drloc.relative_constraint_l1
            elif args.TRAIN.DRLOC_MODE == "ce":  # cross entropy constraint
                reld_criterion = Drloc.relative_constraint_ce
            elif args.TRAIN.DRLOC_MODE == "cbr":  # cycle-back regression constaint: https://arxiv.org/pdf/1904.07846.pdf
                reld_criterion = Drloc.relative_constraint_cbr
            else:
                raise NotImplementedError("We only support l1, ce and cbr now.")

            loss_drloc = 0.0
            for deltaxy, drloc, plane_size in zip(outs.deltaxy, outs.drloc, outs.plz):
                loss_drloc += reld_criterion(deltaxy, drloc) * lambda_drloc
            all_losses.drloc = loss_drloc.item()
            loss += loss_drloc

        return loss, all_losses

    def cal_selfsupervised_loss_3d(self, outs, lambda_drloc=0.0):
        loss, all_losses = 0.0, Munch()

        reld_criterion = Drloc.relative_constraint_l1

        loss_drloc = 0.0
        for deltaxy, drloc, plane_size in zip(outs.deltaxy, outs.drloc, outs.plz):
            loss_drloc += reld_criterion(deltaxy, drloc) * lambda_drloc
        all_losses.drloc = loss_drloc.item()
        loss += loss_drloc

        return loss, all_losses

class MILossGaussian(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(MILossGaussian, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y, x_mask=None, y_mask=None, src=None):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))

        Returns:
            (Normalise)MI: (scalar)
        """
        # filter the images during the loss to compute loss only within the brain
        if x_mask is not None:
            x = x * x_mask
        if y_mask is not None:
            y = y * y_mask

        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)
import torch
import torch.nn.functional as F

class SSIM:
    """
    Structural Similarity Index (SSIM) Loss.
    """

    def __init__(self, window_size=11, max_val=1.0):
        self.window_size = window_size
        self.max_val = max_val
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        kernel = torch.arange(window_size).float() - window_size // 2
        kernel = torch.exp(-kernel**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        kernel_2D = torch.outer(kernel, kernel)  # 正确创建二维高斯核
        return kernel_2D / kernel_2D.sum()

    def create_window(self, window_size, channel):
        _2D_window = self.gaussian_window(window_size, 1.5).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, y_true, y_pred):
        mu_x = F.conv2d(y_true, self.window, padding=self.window_size//2, groups=self.channel)
        mu_y = F.conv2d(y_pred, self.window, padding=self.window_size//2, groups=self.channel)

        sigma_x = F.conv2d(y_true * y_true, self.window, padding=self.window_size//2, groups=self.channel) - mu_x.pow(2)
        sigma_y = F.conv2d(y_pred * y_pred, self.window, padding=self.window_size//2, groups=self.channel) - mu_y.pow(2)
        sigma_xy = F.conv2d(y_true * y_pred, self.window, padding=self.window_size//2, groups=self.channel) - mu_x * mu_y

        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

    def loss(self, y_true, y_pred):
        return 1 - self.ssim(y_true, y_pred)


class SSIM3D:
    """
    3D Structural Similarity Index (SSIM) for volumetric data.
    """

    def __init__(self, window_size=3, max_val=1.0, device='cuda'):
        self.window_size = window_size
        self.max_val = max_val
        self.channel = 1
        self.device = device  # 添加设备属性
        self.window = self.create_window(window_size, self.channel).to(self.device)  # 确保窗口在正确的设备上

    def gaussian_window(self, window_size, sigma):
        kernel = torch.arange(window_size).float() - window_size // 2
        kernel = torch.exp(-kernel**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        kernel_3D = torch.einsum('i,j,k->ijk', kernel, kernel, kernel)
        return kernel_3D / kernel_3D.sum()

    def create_window(self, window_size, channel):
        _3D_window = self.gaussian_window(window_size, 1).unsqueeze(0).unsqueeze(0)
        window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window

    def ssim(self, y_true, y_pred):
        # 确保输入张量也在正确的设备上
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # 计算均值 mu_x 和 mu_y
        mu_x = F.conv3d(y_true, self.window, padding=self.window_size//2, groups=self.channel)
        mu_y = F.conv3d(y_pred, self.window, padding=self.window_size//2, groups=self.channel)

        # 计算方差和协方差
        sigma_x = F.conv3d(y_true * y_true, self.window, padding=self.window_size//2, groups=self.channel) - mu_x.pow(2)
        sigma_y = F.conv3d(y_pred * y_pred, self.window, padding=self.window_size//2, groups=self.channel) - mu_y.pow(2)
        sigma_xy = F.conv3d(y_true * y_pred, self.window, padding=self.window_size//2, groups=self.channel) - mu_x * mu_y

        # SSIM 计算公式
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

    def loss(self, y_true, y_pred):
        return 1 - self.ssim(y_true, y_pred)




    