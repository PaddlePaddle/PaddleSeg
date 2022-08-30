# [Multi-Atlas Labeling](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480/)
Multi-atlas labeling has proven to be an effective paradigm for creating segmentation algorithms from training data. These approaches have been extraordinarily successful for brain and cranial structures (e.g., our prior MICCAI workshops: MLSF’11, MAL’12, SATA’13). After the original challenges closed, the data continue to drive scientific innovation; 144 groups have registered for the 2012 challenge (brain only) and 115 groups for the 2013 challenge (brain/heart/canine leg). However, innovation in application outside of the head and to soft tissues has been more limited. This workshop will provide a snapshot of the current progress in the field through extended discussions and provide researchers an opportunity to characterize their methods on a newly created and released standardized dataset of abdominal anatomy on clinically acquired CT. The datasets will be freely available both during and after the challenge.
## Performance

### TransUnet
> Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for volumetric medical image segmentation." In 2016 fourth international conference on 3D vision (3DV), pp. 565-571. IEEE, 2016.

| Backbone | Resolution | lr | Training Iters | Dice   | Links             |
| --- | --- | --- | --- |--------|-------------------|
| R50-ViT-B_16 | 224x224 | 1e-2 | 13950 | 79.58% | [model]() [log]() |
