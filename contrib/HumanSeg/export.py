import models
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for exporting',
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the export model',
        type=str,
        default='./output/export')
    return parser.parse_args()


def export(args):
    model = models.load_model(args.model_dir)
    model.export_inference_model(args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    export(args)
