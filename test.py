from __future__ import annotations
# python test.py models/RRDB_ESRGAN_x4_old_arch.pth --cpu
# [Universal Models]
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/countryroads_ezogaming_4x_wtfpl_esrgan_universal_model/4x_CountryRoads_377000_G.pth --cpu
# [Realistic Photos]
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/4x_valar_musl_4x_cc0_esrganplus_realistic_photos/4x_Valar_v1.pth --cpu
# [anime]
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/600k_650k_boorugan_tal_4x_wtfpl_esrgan_anime/4x_BooruGan_600k.pth --cpu
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/600k_650k_boorugan_tal_4x_wtfpl_esrgan_anime/4x_BooruGan_650k.pth --cpu
# [manga]
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/fsmangav2_jacob__4x_cc_by-nc-sa_4_0_esrgan_manga/4xFSMangaV2.pth --cpu
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/mangascalev3_bunzeroplusplus_2x_cc_by_nc_sa_4_0_esrganplus_manga/2x_MangaScaleV3.pth --cpu
# [text]
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/nmkd_typescale_nmkd_8x_wtfpl_esrgan_text_2020_11_04/8x_NMKD-Typescale_175k.pth --cpu
# [cartoons]
# python test.py ~/dev/universityofprofessorex/ESRGAN/models/8x_boymebob_redux_joey_8x_cc_by_nc_sa_4_0_esrganplus_joey_s_fork_efonte_fork_or_innfer_required_to_use_animation_2021_08_19_upscaling_cartoons/8x_BoyMeBob-Redux/8x_BoyMeBob-Redux_200000_G.pth --cpu
import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import argparse
import pathlib

import builtins
# import collections
import inspect
import itertools
from os import PathLike, fspath, path
import re
import sys

from typing import TYPE_CHECKING, Optional, Sequence, Type, TypeVar

import numpy as np

if TYPE_CHECKING:
    import packaging.version


ROOT_DIR = path.dirname(path.dirname(__file__))

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # noqa



T = TypeVar("T", str, Sequence[str])


def abspath_or_url(relpath: T) -> T:
    """Utility function that normalizes paths or a sequence thereof.

    Expands user directory and converts relpaths to abspaths... but ignores
    URLS that begin with "http", "ftp", or "file".

    Parameters
    ----------
    relpath : str or list or tuple
        A path, or list or tuple of paths.

    Returns
    -------
    abspath : str or list or tuple
        An absolute path, or list or tuple of absolute paths (same type as
        input).
    """
    from urllib.parse import urlparse

    if isinstance(relpath, (tuple, list)):
        return type(relpath)(abspath_or_url(p) for p in relpath)

    if isinstance(relpath, (str, PathLike)):
        relpath = fspath(relpath)
        urlp = urlparse(relpath)
        if urlp.scheme and urlp.netloc:
            return relpath
        return path.abspath(path.expanduser(relpath))

    raise TypeError("Argument must be a string, PathLike, or sequence thereof")

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--input', default='LR', help='Input folder')
parser.add_argument('--output', default='results', help='Output folder')
parser.add_argument('--cpu', action='store_true', help='Use CPU instead of CUDA')
args = parser.parse_args()

if not os.path.exists(args.model):
    print('Error: Model [{:s}] does not exist.'.format(args.model))
    sys.exit(1)
elif not os.path.exists(args.input):
    print('Error: Folder [{:s}] does not exist.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.input):
    print('Error: Folder [{:s}] is a file.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.output):
    print('Error: Folder [{:s}] is a file.'.format(args.output))
    sys.exit(1)
elif not os.path.exists(args.output):
    os.mkdir(args.output)

# import bpdb
# bpdb.set_trace()

model_path = args.model
device = torch.device('cpu' if args.cpu else 'cuda')

test_img_folder = os.path.join(os.path.normpath(args.input), '*')
output_folder = os.path.normpath(args.output)

state_dict = torch.load(model_path)

if 'conv_first.weight' in state_dict:
    print('Error: Attempted to load a new-format model')
    sys.exit(1)

# extract model information
scale2 = 0
max_part = 0
for part in list(state_dict):
    parts = part.split('.')
    n_parts = len(parts)
    if n_parts == 5 and parts[2] == 'sub':
        nb = int(parts[3])
    elif n_parts == 3:
        part_num = int(parts[1])
        if part_num > 6 and parts[2] == 'weight':
            scale2 += 1
        if part_num > max_part:
            max_part = part_num
            out_nc = state_dict[part].shape[0]
upscale = 2 ** scale2
in_nc = state_dict['model.0.weight'].shape[1]
nf = state_dict['model.0.weight'].shape[0]

model = arch.RRDB_Net(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(state_dict, strict=True)
del state_dict
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    if os.path.isdir(path): # skip directories
        continue
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img * 1. / np.iinfo(img.dtype).max

    if img.ndim == 2:
        img = np.tile(np.expand_dims(img, axis=2), (1, 1, min(in_nc, 3)))
    if img.shape[2] > in_nc: # remove extra channels
        if in_nc != 3 or img.shape[2] != 4 or img[:, :, 3].min() < 1:
            print('Warning: Truncating image channels')
        img = img[:, :, :in_nc]
    elif img.shape[2] == 3 and in_nc == 4: # pad with solid alpha channel
        img = np.dstack((img, np.full(img.shape[:-1], 1.)))

    if img.shape[2] == 3:
        img = img[:, :, [2, 1, 0]]
    elif img.shape[2] == 4:
        img = img[:, :, [2, 1, 0, 3]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    if output.shape[0] == 3:
        output = output[[2, 1, 0], :, :]
    elif output.shape[0] == 4:
        output = output[[2, 1, 0, 3], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(os.path.join(output_folder, '{:s}_rlt.png'.format(base)), output)
