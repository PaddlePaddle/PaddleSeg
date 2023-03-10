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
from .convert_to_decathlon import convert_to_decathlon
from .integrity_checks import verify_dataset_integrity
from .image_crop import crop
from .dataset_analyzer import DatasetAnalyzer
from .preprocessing import GenericPreprocessor, PreprocessorFor2D, get_lowres_axis, get_do_separate_z, resize_segmentation, resample_data_or_seg
from .experiment_utils import *
from .experiment_planner import ExperimentPlanner2D_v21, ExperimentPlanner3D_v21
from .file_and_folder_operations import *
from .path_utils import join_paths
