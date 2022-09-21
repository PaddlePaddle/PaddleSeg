## Traning

You can start the training program, such as the following command:

```shell
python train.py --config configs/swinsunet/swinunet_abdomen_224_224_1_14k_5e-2.yml --do_eval --save_interval 1000 --has_dataset_json False --is_save_data False --num_workers 4 --log_iters 10 --use_vdl --seed 998
```

## Performance

### SwinUnet
> [Hu Cao](https://arxiv.org/search/eess?searchtype=author&query=Cao%2C+H), [Yueyue Wang](https://arxiv.org/search/eess?searchtype=author&query=Wang%2C+Y), [Joy Chen](https://arxiv.org/search/eess?searchtype=author&query=Chen%2C+J), [Dongsheng Jiang](https://arxiv.org/search/eess?searchtype=author&query=Jiang%2C+D), [Xiaopeng Zhang](https://arxiv.org/search/eess?searchtype=author&query=Zhang%2C+X), [Qi Tian](https://arxiv.org/search/eess?searchtype=author&query=Tian%2C+Q), [Manning Wang](https://arxiv.org/search/eess?searchtype=author&query=Wang%2C+M). "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation." arXiv preprint arXiv:2105.05537, 2021.

| Backbone | Resolution | lr | Training Iters | Dice |  Links |
| --- | --- | --- | --- | --- | --- |
| SwinTransformer-tinyer | 224x224 | 5e-2 | 14000 | 79.97% | [model]() \| [log]() \| [vdl]() |


#### Inference helper User Guide

1. Since the shape of the input and output data of the SwinUnet network is different from other networks, in order to make it compatible with the inference program, it is necessary to preprocess and post-process the data of the network. This part of the work is done by InferenceHelper.
2. InferenceHelper is an abstract base class that contains two methods, preprocess and postprocess. If you need to add a new InferenceHelper to your own network, you need to customize the class in the medicalseg/inference_helpers package and inherit this base InferenceHelper class.
```Python
class SwinUNetInferenceHelper(InferenceHelper):
```

3. Medical maintains a INFERENCE_HELPERS variable of type ComponentManager. You can add your custom InferenceHelper through the add_component method.
```Python
@manager.INFERENCE_HELPERS.add_component
class SwinUNetInferenceHelper(InferenceHelper):
```
Also you need to import your class in the \_\_init\_\_.py file of the inference_helper package.
```Python
# in medicalseg/inference_helpers/__init__.py file

from .transunet_inference_helper import SwinUNetInferenceHelper
```

4. You also need to implement preprocess and postprocess methods suitable for your own network, such as the following:

```Python
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

```Python
# in yml file of configs

export:
  inference_helper:
    type: SwinUNetInferenceHelper
```
