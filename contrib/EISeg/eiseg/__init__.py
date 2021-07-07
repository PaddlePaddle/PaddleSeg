import sys
import os.path as osp

pjpath = osp.dirname(osp.realpath(__file__))
sys.path.append(pjpath)


from run import main
from models import models


__APPNAME__ = "EISeg"
