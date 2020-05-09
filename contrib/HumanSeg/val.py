import argparse
from datasets.dataset import Dataset
import transforms
import models


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for evaluating',
        type=str,
        default='output/best_model')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='The root directory of dataset',
        type=str)
    parser.add_argument(
        '--val_list',
        dest='val_list',
        help='Val list file of dataset',
        type=str,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=128)
    return parser.parse_args()


def evaluate(args):
    eval_transforms = transforms.Compose(
        [transforms.Resize((192, 192)),
         transforms.Normalize()])

    eval_dataset = Dataset(
        data_dir=args.data_dir,
        file_list=args.val_list,
        transforms=eval_transforms,
        num_workers='auto',
        buffer_size=100,
        parallel_method='thread',
        shuffle=False)

    model = models.load_model(args.model_dir)
    model.evaluate(eval_dataset, args.batch_size)


if __name__ == '__main__':
    args = parse_args()

    evaluate(args)
