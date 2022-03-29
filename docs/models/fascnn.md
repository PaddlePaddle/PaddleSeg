### Fast-SCNN

Fast-SCNN is a real-time oriented semantic segmentation network. Based on the dual-branch structure, depthwise separable convolution and inverted-residual modules are extensively used, and feature fusion is used to construct a Pyramid Pooling Module to fuse context information. This enables Fast-SCNN to learn rich details while remaining efficient. The biggest feature of Fast-SCNN is "small and fast", that is, the model only needs small FLOPs during inference calculation, and can quickly infer a good result. The entire network structure is as follows:

![img](./images/Fast-SCNN.png)

<div align = "center">Fast-SCNN</div>

For details, please refer to[Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502).
