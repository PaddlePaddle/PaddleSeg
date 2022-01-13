import paddle
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from data.datasets import *
from model.losses import *
from data.transforms import *
#from isegm.engine.trainer import ISTrainer
from model.metrics import AdaptiveIoU
from data.points_sampler import MultiPointSampler
from model.initializer import XavierGluon

from model.is_hrnet_model import HRNetModel
from model.is_deeplab_model import DeeplabModel