import argparse
from datasets.dataset import Dataset
from models import HumanSegMobile, HumanSegLite, HumanSegServer
import transforms

MODEL_TYPE = ['HumanSegMobile', 'HumanSegLite', 'HumanSegServer']


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_type',
        dest='model_type',
        help=
        "Model type for traing, which is one of ('HumanSegMobile', 'HumanSegLite', 'HumanSegServer')",
        type=str,
        default='HumanSegMobile')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='The root directory of dataset',
        type=str)
    parser.add_argument(
        '--train_list',
        dest='train_list',
        help='Train list file of dataset',
        type=str)
    parser.add_argument(
        '--val_list',
        dest='val_list',
        help='Val list file of dataset',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output/quant_train')
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        type=int,
        default=2)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='Number epochs for training',
        type=int,
        default=2)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=128)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=0.001)
    parser.add_argument(
        '--pretrained_weights',
        dest='pretrained_weights',
        help='The model path for quant',
        type=str,
        default=None)
    parser.add_argument(
        '--save_interval_epochs',
        dest='save_interval_epochs',
        help='The interval epochs for save a model snapshot',
        type=int,
        default=1)

    return parser.parse_args()


def train(args):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((192, 192)),
        transforms.Normalize()
    ])

    eval_transforms = transforms.Compose(
        [transforms.Resize((192, 192)),
         transforms.Normalize()])

    train_dataset = Dataset(
        data_dir=args.data_dir,
        file_list=args.train_list,
        transforms=train_transforms,
        num_workers='auto',
        buffer_size=100,
        parallel_method='thread',
        shuffle=True)

    eval_dataset = None
    if args.val_list is not None:
        eval_dataset = Dataset(
            data_dir=args.data_dir,
            file_list=args.val_list,
            transforms=eval_transforms,
            num_workers='auto',
            buffer_size=100,
            parallel_method='thread',
            shuffle=False)

    model = eval(args.model_type)(num_classes=2)

    model.train(
        num_epochs=args.num_epochs,
        train_dataset=train_dataset,
        train_batch_size=args.batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=args.save_interval_epochs,
        save_dir=args.save_dir,
        pretrained_weights=args.pretrained_weights,
        learning_rate=args.learning_rate,
        quant=True)


if __name__ == '__main__':
    args = parse_args()

    if args.model_type not in MODEL_TYPE:
        raise ValueError(
            "--model_type: {} is set wrong, it shold be one of ('HumanSegMobile', "
            "'HumanSegLite', 'HumanSegServer')".format(args.model_type))
    train(args)
