import paddle
from paddleseg.models import MscaleOCR
from paddleseg.core.val import evaluate
from paddleseg.datasets import Cityscapes
import paddleseg.transforms as T
import numpy as np
model = MscaleOCR(num_classes=19)
state_dict = paddle.load('pretrained/best_checkpoint_86.98_PSA_p.pdparams')
val_transforms = [
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]

# 构建验证集
val_dataset = Cityscapes(
    dataset_root='data/cityscapes',
    transforms=val_transforms,
    mode='val'
)
model.set_state_dict(state_dict)

batch_sampler = paddle.io.DistributedBatchSampler(
    val_dataset, batch_size=1, shuffle=False, drop_last=False)
loader = paddle.io.DataLoader(
    val_dataset,
    batch_sampler=batch_sampler,
    num_workers=4,
    return_list=True, )

import matplotlib.pyplot as plt 
i = 0
for data in loader:
    if i ==6:
        break
    else:
        i += 1
    with paddle.no_grad(): 
        img = model(data[0])
    img = paddle.argmax(img[0],axis=1)
    img = img[0].numpy()
    gt = data[1].numpy()[0][0]
    img[np.where(gt==255)] = 255
    plt.figure(figsize=(10,4),dpi=240)
    plt.subplot(131)
    plt.title('Image')
    plt.axis('off')
    plt.imshow(data[0].numpy()[0].transpose(1,2,0))
    plt.subplot(132)
    plt.title('Predict')
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(133)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.imshow(gt)
    plt.savefig('测试/'+str(i)+'.png')
    plt.show()