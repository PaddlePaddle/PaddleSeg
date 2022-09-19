# [Multi-Atlas Labeling](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480/)
Multi-atlas labeling has proven to be an effective paradigm for creating segmentation algorithms from training data. These approaches have been extraordinarily successful for brain and cranial structures (e.g., our prior MICCAI workshops: MLSF’11, MAL’12, SATA’13). After the original challenges closed, the data continue to drive scientific innovation; 144 groups have registered for the 2012 challenge (brain only) and 115 groups for the 2013 challenge (brain/heart/canine leg). However, innovation in application outside of the head and to soft tissues has been more limited. This workshop will provide a snapshot of the current progress in the field through extended discussions and provide researchers an opportunity to characterize their methods on a newly created and released standardized dataset of abdominal anatomy on clinically acquired CT. The datasets will be freely available both during and after the challenge.
## Performance

### TransUnet
> Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation." arXiv preprint arXiv:2102.04306, 2021.

| Backbone | Resolution | lr | Training Iters | Dice |  Links |
| --- | --- | --- | --- | --- | --- |
| R50-ViT-B_16 | 224x224 | 1e-2 | 13950 | 81.05% | [model](https://paddleseg.bj.bcebos.com/paddleseg3d/synapse/abdomen/transunet_abdomen_224_224_1_14k_1e-2/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/paddleseg3d/synapse/abdomen/transunet_abdomen_224_224_1_14k_1e-2/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=d933d970394436aa6969c9c00cf8a6da) |


#### Inference helper User Guide

1. Since the shape of the input and output data of the TransUnet network is different from other networks, in order to make it compatible with the inference program, it is necessary to preprocess and post-process the data of the network. This part of the work is done by InferenceHelper.
2. InferenceHelper is an abstract base class that contains two methods, preprocess and postprocess. If you need to add a new InferenceHelper to your own network, you need to customize the class in the medicalseg/inference_helpers package and inherit this base InferenceHelper class.
```
class TransUNetInferenceHelper(InferenceHelper):
```

3. Medical maintains a INFERENCE_HELPERS variable of type ComponentManager. You can add your custom InferenceHelper through the add_component method.
```
@manager.INFERENCE_HELPERS.add_component
class TransUNetInferenceHelper(InferenceHelper):
```
Also you need to import your class in the \_\_init\_\_.py file of the inference_helper package.
```
# in medicalseg/inference_helpers/__init__.py file

from .transunet_inference_helper import TransUNetInferenceHelper
```

4. You also need to implement preprocess and postprocess methods suitable for your own network, such as the following:

```
    def preprocess(self, cfg, imgs_path, batch_size, batch_id):
        for img in imgs_path[batch_id:batch_id + batch_size]:
            im_list = []
            imgs = np.load(img)
            imgs = imgs[:, np.newaxis, :, :]
            for i in range(imgs.shape[0]):
                im = imgs[i]
                im = cfg.transforms(im)[0]
                im_list.append(im)
            img = np.concatenate(im_list)
        return img

    def postprocess(self, results):
        results = np.argmax(results, axis=1)
        results = results[np.newaxis, :, :, :, :]
        return results
```

5. Finally you can reference your Inference helper in your yaml configuration file. This way
your InferenceHelper is automatically called in the inference script.

```
# in yml file of configs

export:
  inference_helper:
    type: TransUNetInferenceHelper
```
