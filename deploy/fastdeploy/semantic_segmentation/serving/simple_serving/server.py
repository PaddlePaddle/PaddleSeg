import fastdeploy as fd
from fastdeploy.serving.server import SimpleServer
import os
import logging

logging.getLogger().setLevel(logging.INFO)

# Configurations
model_dir = 'PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer'
device = 'cpu'
use_trt = False

# Prepare model
model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "deploy.yaml")

# Setup runtime option to select hardware, backend, etc.
option = fd.RuntimeOption()
if device.lower() == 'gpu':
    option.use_gpu()
if use_trt:
    option.use_trt_backend()
    option.set_trt_cache_file('pp_lite_seg.trt')

# Create model instance
model_instance = fd.vision.segmentation.PaddleSegModel(
    model_file=model_file,
    params_file=params_file,
    config_file=config_file,
    runtime_option=option)

# Create server, setup REST API
app = SimpleServer()
app.register(
    task_name="fd/ppliteseg",
    model_handler=fd.serving.handler.VisionModelHandler,
    predictor=model_instance)
