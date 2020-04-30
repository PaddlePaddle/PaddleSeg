import HumanSeg
from HumanSeg.datasets.dataset import Dataset
from HumanSeg.transforms import transforms

eval_transforms = transforms.Compose(
    [transforms.Resize((192, 192)),
     transforms.Normalize()])

data_dir = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly'
val_list = '/ssd1/chenguowei01/dataset/humanseg/supervise.ly/val.txt'

eval_dataset = Dataset(
    data_dir=data_dir,
    file_list=val_list,
    transforms=eval_transforms,
    num_workers='auto',
    buffer_size=100,
    parallel_method='thread',
    shuffle=False)

eval_dataset = Dataset(
    data_dir=data_dir,
    file_list=val_list,
    transforms=eval_transforms,
    num_workers='auto',
    buffer_size=100,
    parallel_method='thread',
    shuffle=False)

model = HumanSeg.models.load_model('output/best_model')
model.evaluate(eval_dataset, 2)
