import os
from easydict import EasyDict as edict
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dirs in ['./save']:
    if os.path.exists(dirs):
        __C.GLOBAL.MODEL_SAVE_DIR = dirs
assert __C.GLOBAL.MODEL_SAVE_DIR is not None

__C.DATA = edict()
__C.DATA.IMG_SIZE = 480


