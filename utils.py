import os
import os.path as osp
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
from torch import nn
import jiwer

import matplotlib.pylab as plt
import functools
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from einops import rearrange
from scipy import ndimage
from torch.special import gammaln


def calc_wer(target, pred, ignore_indexes=[0]):
    target_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(target)))))
    pred_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(pred)))))
    target_str = ' '.join(target_chars)
    pred_str = ' '.join(pred_chars)
    error = jiwer.wer(target_str, pred_str)
    return error

def drop_duplicated(chars):
    ret_chars = [chars[0]]
    for prev, curr in zip(chars[:-1], chars[1:]):
        if prev != curr:
            ret_chars.append(curr)
    return ret_chars

# def build_criterion(critic_params={}):

#     criterion = {
#         "ce": nn.CrossEntropyLoss(ignore_index=-1),
#         "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
#         "hinge": nn.HingeEmbeddingLoss(margin=critic_params.get('hinge', {}).get("margin", 1.0))
#     }
#     return criterion

def build_criterion(critic_params={}):
    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1),
        "ctc": torch.nn.CTCLoss(**critic_params.get('ctc', {})),
    }
    return criterion



def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list


def plot_image(image):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(image, aspect="auto", origin="lower",
                   interpolation='none')

    fig.canvas.draw()
    plt.close()

    return fig



class PartialConv1d(torch.nn.Conv1d):
    """
    Zero padding creates a unique identifier for where the edge of the data is, such that the model can almost always identify
    exactly where it is relative to either edge given a sufficient receptive field. Partial padding goes to some lengths to remove 
    this affect.
    """

    __constants__ = ['slide_winsize']
    slide_winsize: float

    def __init__(self, *args, **kwargs):
        super(PartialConv1d, self).__init__(*args, **kwargs)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.register_buffer("weight_maskUpdater", weight_maskUpdater, persistent=False)
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2]

    def forward(self, input, mask_in):
        if mask_in is None:
            mask = torch.ones(1, 1, input.shape[2], dtype=input.dtype, device=input.device)
        else:
            mask = mask_in
            input = torch.mul(input, mask)
        with torch.no_grad():
            update_mask = F.conv1d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            update_mask_filled = torch.masked_fill(update_mask, update_mask == 0, self.slide_winsize)
            mask_ratio = self.slide_winsize / update_mask_filled
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio, update_mask)

        raw_out = self._conv_forward(input, self.weight, self.bias)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            output = torch.mul(raw_out - bias_view, mask_ratio) + bias_view
            output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        return output


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    __constants__ = ['use_partial_padding']
    use_partial_padding: bool

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain='linear',
        use_partial_padding=False,
        use_weight_norm=False,
        norm_fn=None,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.use_partial_padding = use_partial_padding
        conv_fn = torch.nn.Conv1d
        if use_partial_padding:
            conv_fn = PartialConv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if use_weight_norm:
            self.conv = torch.nn.utils.weight_norm(self.conv)
        if norm_fn is not None:
            self.norm = norm_fn(out_channels, affine=True)
        else:
            self.norm = None

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            ret = self.conv(signal, mask)
            if self.norm is not None:
                ret = self.norm(ret, mask)
        else:
            if mask is not None:
                signal = signal.mul(mask)
            ret = self.conv(signal)
            if self.norm is not None:
                ret = self.norm(ret)

        # if self.is_adapter_available():
        #     ret = self.forward_enabled_adapters(ret.transpose(1, 2)).transpose(1, 2)

        return ret



class BetaBinomialInterpolator:
    """
    This module calculates alignment prior matrices (based on beta-binomial distribution) using cached popular sizes and image interpolation.
    The implementation is taken from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/data_function.py
    """

    def __init__(self, round_mel_len_to=50, round_text_len_to=10, cache_size=500, scaling_factor: float = 1.0):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        cached_func = lambda x, y: beta_binomial_prior_distribution(x, y, scaling_factor=scaling_factor)
        self.bank = functools.lru_cache(maxsize=cache_size)(cached_func)

    @staticmethod
    def round(val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = BetaBinomialInterpolator.round(w, to=self.round_mel_len_to)
        bh = BetaBinomialInterpolator.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def general_padding(item, item_len, max_len, pad_value=0):
    if item_len < max_len:
        item = torch.nn.functional.pad(item, (0, max_len - item_len), value=pad_value)
    return item


def stack_tensors(tensors: List[torch.Tensor], max_lens: List[int], pad_value: float = 0.0) -> torch.Tensor:
    """
    Create batch by stacking input tensor list along the time axes.

    Args:
        tensors: List of tensors to pad and stack
        max_lens: List of lengths to pad each axis to, starting with the last axis
        pad_value: Value for padding

    Returns:
        Padded and stacked tensor.
    """
    padded_tensors = []
    for tensor in tensors:
        padding = []
        for i, max_len in enumerate(max_lens, 1):
            padding += [0, max_len - tensor.shape[-i]]

        padded_tensor = torch.nn.functional.pad(tensor, pad=padding, value=pad_value)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor


def logbeta(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logcombinations(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def logbetabinom(n, a, b, x):
    return logcombinations(n, x) + logbeta(x + a, n - x + b) - logbeta(a, b)


def beta_binomial_prior_distribution(phoneme_count: int, mel_count: int, scaling_factor: float = 1.0) -> np.array:
    x = rearrange(torch.arange(0, phoneme_count), "b -> 1 b")
    y = rearrange(torch.arange(1, mel_count + 1), "b -> b 1")
    a = scaling_factor * y
    b = scaling_factor * (mel_count + 1 - y)
    n = torch.FloatTensor([phoneme_count - 1])

    return logbetabinom(n, a, b, x).exp().numpy()


# example : attn_prior = (torch.from_numpy(beta_binomial_interpolator(spect_len.item(), text_len.item())).unsqueeze(0).to(text.device))
