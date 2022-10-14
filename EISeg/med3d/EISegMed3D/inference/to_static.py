import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "./.."))

import yaml
import paddle
import paddle.nn as nn
from paddleseg.utils import logger
from inference.models import VNetModel


def main():
    model = VNetModel(
        elu=False,
        in_channels=1,
        num_classes=1,
        pretrained="/ssd2/tangshiyu/Code/EISeg-3D/experiments/3D_interseg/mrispineseg_vnet/174/checkpoints/049.pdparams",  # "pretrained_models/vnet_model.pdparams",
        kernel_size=[[2, 2, 4], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        stride_size=[[2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2]],
        with_aux_output=False,
        use_leaky_relu=True,
        use_rgb_conv=False,
        use_disks=True,
        norm_radius=2,
        with_prev_mask=True, )

    model.set_dict(paddle.load("model_checkpoints/039.pdparams"))
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # on or off did not change
    model.eval()
    print("Loaded trained params of model successfully")

    new_net = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 1, None, None, None], dtype="float32"),
            paddle.static.InputSpec(
                shape=[None, 3, None, None, None],
                dtype="float32"),  # 16, 48, 4
        ], )

    paddle.jit.save(new_net, "output_cpu/static_Vnet_model")

    yml_file = os.path.join("output_cpu/static_VNet_model", "vnet_deploy.yaml")
    with open(yml_file, "w") as file:
        data = {
            "Deploy": {
                "model": "static_Vnet_model.pdmodel",
                "params": "static_Vnet_model.pdiparams"
            }
        }
        yaml.dump(data, file)

    logger.info("Model is saved in {}".format("output_cpu"))


if __name__ == "__main__":
    main()
