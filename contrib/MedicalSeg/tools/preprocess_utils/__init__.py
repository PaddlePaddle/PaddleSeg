import yaml
import codecs
from . import global_var
# Import global_val then everywhere else can change/use the global dict
with codecs.open('tools/preprocess_globals.yml', 'r', 'utf-8') as file:
    dic = yaml.load(file, Loader=yaml.FullLoader)
global_var.init()
if dic['use_gpu']:
    global_var.set_value('USE_GPU', True)
else:
    global_var.set_value('USE_GPU', False)

from .values import *
from .uncompress import uncompressor
from .geometry import *
from .load_image import *
from .dataset_json import parse_msd_basic_info
