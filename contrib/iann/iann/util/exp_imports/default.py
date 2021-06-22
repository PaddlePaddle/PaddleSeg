import paddle
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from data.datasets import *
from models.losses import *
from data.transforms import *
#from isegm.engine.trainer import ISTrainer
from models.metrics import AdaptiveIoU
from data.points_sampler import MultiPointSampler
from models.initializer import XavierGluon

from models.is_hrnet_model import HRNetModel
from models.is_deeplab_model import DeeplabModel