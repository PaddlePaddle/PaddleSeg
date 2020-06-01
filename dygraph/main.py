from datasets import Dataset
import transforms
import paddle.fluid as fluid
from models import UNet

data_dir = '/ssd1/chenguowei01/dataset/optic_disc_seg'
train_list = '/ssd1/chenguowei01/dataset/optic_disc_seg/train_list.txt'
val_list = '/ssd1/chenguowei01/dataset/optic_disc_seg/val_list.txt'
img_file = data_dir + '/JPEGImages/H0005.jpg'

train_transforms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

train_dataset = Dataset(
    data_dir=data_dir,
    file_list=train_list,
    transforms=train_transforms,
    num_workers='auto',
    buffer_size=100,
    parallel_method='thread',
    shuffle=True)

eval_transforms = transforms.Compose(
    [transforms.Resize((192, 192)),
     transforms.Normalize()])

eval_dataset = Dataset(
    data_dir=data_dir,
    file_list=val_list,
    transforms=eval_transforms,
    num_workers='auto',
    buffer_size=100,
    parallel_method='thread',
    shuffle=True)

model = UNet(num_classes=2)
with fluid.dygraph.guard(model.places):
    model.build_model()
    #model.load_model('output/epoch_10/')
    model.train(
        num_epochs=10, train_dataset=train_dataset, eval_dataset=eval_dataset)
    model.evaluate(eval_dataset)
    model.predict(img_file, eval_transforms)
