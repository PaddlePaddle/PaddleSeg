import argparse
from datasets.dataset import Dataset
import transforms
import models


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for quant',
        type=str,
        default='output/best_model')
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=1)
    parser.add_argument(
        '--batch_nums',
        dest='batch_nums',
        help='Batch number for quant',
        type=int,
        default=10)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='the root directory of dataset',
        type=str)
    parser.add_argument(
        '--quant_list',
        dest='quant_list',
        help=
        'Image file list for model quantization, it can be vat.txt or train.txt',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the quant model',
        type=str,
        default='./output/quant_offline')
    return parser.parse_args()


def evaluate(args):
    eval_transforms = transforms.Compose(
        [transforms.Resize((192, 192)),
         transforms.Normalize()])

    eval_dataset = Dataset(
        data_dir=args.data_dir,
        file_list=args.quant_list,
        transforms=eval_transforms,
        num_workers='auto',
        buffer_size=100,
        parallel_method='thread',
        shuffle=False)

    model = models.load_model(args.model_dir)
    model.export_quant_model(
        dataset=eval_dataset,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        batch_nums=args.batch_nums)


if __name__ == '__main__':
    args = parse_args()

    evaluate(args)
