import os

import numpy as np
import torch
import torch.jit
import torch.nn.functional as F
import cv2


@torch.jit.script
def psnr(X, Y, data_range: float):
    """
    Peak Signal to Noise Ratio
    我们取50，如果完美恢复
    """
    mse = torch.mean((X - Y) ** 2)
    if mse == 0:
        return torch.tensor(50.)
    else:
        return 10 * torch.log10(data_range ** 2 / mse)


class PSNR(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'avg']

    def __init__(self, data_range=1., avg=True):
        super().__init__()
        self.data_range = data_range
        self.avg = avg

    @torch.jit.script_method
    def forward(self, X, Y):
        r = psnr(X, Y, self.data_range)
        if self.avg:
            return r.mean()
        else:
            return r


@torch.jit.script
def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


@torch.jit.script
def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


@torch.jit.script
def ssim(X, Y, window, data_range: float, use_padding: bool = False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return: (N,1)
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


@torch.jit.script
def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool = False):
    '''
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :return: (N,1)
    '''
    levels = weights.shape[0]
    cs_vals = []
    ssim_vals = []
    for _ in range(levels):
        ssim_val, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)
        cs_vals.append(cs)
        ssim_vals.append(ssim_val)
        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, stride=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, stride=2, padding=padding)

    cs_vals = torch.stack(cs_vals, dim=0)
    ms_ssim_val = torch.prod((cs_vals[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_vals[-1] ** weights[-1]), dim=0)
    return ms_ssim_val


class SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding', 'avg']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False, avg=True):
        '''
        Structural Similarity Index
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        :param avg: average between the batch
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding
        self.avg = avg

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)[0]
        if self.avg:
            return r.mean()
        else:
            return r


class MS_SSIM(torch.jit.ScriptModule):
    __constants__ = ['data_range', 'use_padding', 'avg']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=1., channel=3, use_padding=False, weights=None,
                 levels=None, avg=True):
        '''
        Multi-Scale Structural Similarity Index
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param avg: average between the batch
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.avg = avg

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    @torch.jit.script_method
    def forward(self, X, Y):
        r = ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                    use_padding=self.use_padding)
        if self.avg:
            return r.mean()
        else:
            return r


if __name__ == "__main__":
    # 使用时修改ref_dir和test_dir
    # mode有三种，PSNR SSIM MS_SSIM
    # img_mode有两种，HDR LDR    其中HDR格式需要hdr_io_tools.py，且是对HDR图像tonemap以后再测指标而非对luminance直接测！
    # tonemap的参数默认用cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)，要改就到代码里面改

    ref_dir = './gt/'
    test_dir = './results'
    mode = 'PSNR'
    img_mode = 'LDR'
    tmo = None

    if img_mode == 'HDR':
        import hdr_io_tools
        tmo = cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)  # tonemap参数，看情况修改

    if mode == 'MS_SSIM':
        m = 'MS_SSIM'
        metric = MS_SSIM().cuda()
    elif mode == 'SSIM':
        m = 'SSIM'
        metric = SSIM().cuda()
    else:
        m = 'PSNR'
        metric = PSNR().cuda()

    md = {}
    avg = 0
    for f in os.listdir(ref_dir):
        ref_name = os.path.join(ref_dir, f)
        name = f.split('.')[0]
        name_ = name + ''
        # name_ = name

        test_name = None
        image_ref = None
        image_test = None

        if img_mode == 'HDR':
            # for HDR images
            if not (f.endswith('.hdr') or f.endswith('.exr')):
                continue
            if os.path.exists(os.path.join(test_dir, name_ + '.hdr')):
                test_name = os.path.join(test_dir, name_ + '.hdr')
            elif os.path.exists(os.path.join(test_dir, name_ + '.exr')):
                test_name = os.path.join(test_dir, name_ + '.exr')
            else:
                raise Exception("Cannot find " + name_ + ".hdr or .exr")
            # read
            image_ref = cv2.cvtColor(hdr_io_tools.read(ref_name), cv2.COLOR_RGB2BGR).astype('float32')  # (H, W, C)
            image_test = cv2.cvtColor(hdr_io_tools.read(test_name), cv2.COLOR_RGB2BGR).astype('float32')
            # 对HDR图像要先tonemap再算指标(注意要先归一化到[0,1]然后再tonemap)
            image_ref = (image_ref - np.min(image_ref)) / (np.max(image_ref) - np.min(image_ref))
            image_test = (image_test - np.min(image_test)) / (np.max(image_test) - np.min(image_test))
            image_ref = (tmo.process(image_ref) * 255).astype('uint8')
            image_test = (tmo.process(image_test) * 255).astype('uint8')
        else:
            # for LDR images
            if not (f.endswith('.jpg') or f.endswith('.png')):
                continue
            if os.path.exists(os.path.join(test_dir, name_ + '.jpg')):
                test_name = os.path.join(test_dir, name_ + '.jpg')
            elif os.path.exists(os.path.join(test_dir, name_ + '.png')):
                test_name = os.path.join(test_dir, name_ + '.png')
            else:
                raise Exception("Cannot find " + name_ + ".jpg or .png")
            # read

            image_ref = cv2.imread(ref_name)  # (H, W, C)
            image_test = cv2.imread(test_name)
            #image_test =cv2.resize(image_test, (image_ref.shape[1], image_ref.shape[0]))

        # transpose
        image_ref = np.expand_dims(np.transpose(image_ref, (2, 0, 1)), axis=0)  # (1, C, H, W)
        image_test = np.expand_dims(np.transpose(image_test, (2, 0, 1)), axis=0)
        # to tensor
        image_ref_tensor = torch.from_numpy(image_ref).cuda().float()
        image_test_tensor = torch.from_numpy(image_test).cuda().float()
        # scale to [0,1]
        image_ref_tensor /= torch.max(image_ref_tensor)
        image_test_tensor /= torch.max(image_test_tensor)
        with torch.no_grad():
            md[name] = metric(image_test_tensor, image_ref_tensor).item()
        avg += md[name]

    avg /= len(md)
    print(avg)
    with open(test_dir + '_' + m + '.txt', 'w') as fd:
        for i in md:
            fd.write('--------------------\nname: {}\n'.format(i))
            fd.write(m + ': {}\n'.format(md[i]))
        fd.write('\n\n====================\n')
        fd.write('avg: {}\n'.format(avg))
