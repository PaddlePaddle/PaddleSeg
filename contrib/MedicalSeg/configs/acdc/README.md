# [Automated cardiac diagnosis](https://acdc.creatis.insa-lyon.fr/description/databases.html)
The database is made available to participants through two datasets from the dedicated online evaluation website after a personal registration: i) a training dataset of 100 patients along with the corresponding manual references based on the analysis of one clinical expert; ii) a testing dataset composed of 50 new patients, without manual annotations but with the patient information given above. The raw input images are provided through the Nifti format.
## Performance


### nnformer
>   Hong-Yu Zhou, Student Member, IEEE, Jiansen Guo, Yinghao Zhang, Xiaoguang Han, Lequan Yu, Liansheng Wang, Member, IEEE, and Yizhou Yu, Fellow, IEEE

| Backbone | Resolution | lr | Training Iters | Dice | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|
|-|14x160x160|1e-4|250000|91.78%|[model](https://paddleseg.bj.bcebos.com/paddleseg3d/acdc/nnformer_acdc_160_160_14_250k_4e-4/model.pdparams)\| [log](https://paddleseg.bj.bcebos.com/paddleseg3d/acdc/nnformer_acdc_160_160_14_250k_4e-4/train.log)\| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=b9a90b8aba579997a6f088b840a6e96d)|
