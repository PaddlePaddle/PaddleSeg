# coding: utf8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import os.path as osp

dataset_root = "."
DIM_DATASET_ROOT_PATH = osp.join(dataset_root, "Composition-1k")
AM2K_DATASET_ROOT_PATH = osp.join(dataset_root, "AM-2K")
BG20K_DATASET_ROOT_PATH = osp.join(dataset_root, "BG-20k")

# all datasets path dict
DATASET_PATHS_DICT = {
    'DIM':{
	    'TRAIN':{
		    'ROOT_PATH': osp.join(DIM_DATASET_ROOT_PATH, 'train'),
		    'FG_PATH': osp.join(DIM_DATASET_ROOT_PATH, 'train/fg'),
		    'ALPHA_PATH': osp.join(DIM_DATASET_ROOT_PATH, 'train/alpha'),
		    'SAMPLE_NUMBER': 431,
		    'SAMPLE_INTERVAL': 5
			}
		},
    'AM2K':{
	    'TRAIN':{
	    	'ROOT_PATH': osp.join(AM2K_DATASET_ROOT_PATH, 'train'),
	    	'FG_PATH': osp.join(AM2K_DATASET_ROOT_PATH, 'train/fg'),
	    	'ALPHA_PATH': osp.join(AM2K_DATASET_ROOT_PATH, 'train/alpha'), 
	    	'SAMPLE_NUMBER':1800,
	    	'SAMPLE_INTERVAL':2
	    	}
		},
    'BG20K':{
	    'TRAIN': {
	    	'ROOT_PATH': BG20K_DATASET_ROOT_PATH,
	    	'ORIGINAL_PATH': osp.join(BG20K_DATASET_ROOT_PATH, 'train')
	    	},
		},
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='PaddleSeg generate file list on hybrid dataset.'
    )
    parser.add_argument('dataset_root', help='dataset root directory', type=str)
    parser.add_argument(
        '--separator',
        dest='separator',
        help='file list separator',
        default=" ",
        type=str)
    parser.add_argument(
        '--datasets',
        help='the datasets will mix.',
        type=str,
		nargs='*',
        default=['DIM', 'AM-2K'])
    parser.add_argument(
        '--split',
        help='the split of datasets',
        type=str,
        default='train')

    return parser.parse_args()

def get_files(path, shuffle=False):
	"""
	get file paths and filter useless folders.

	Args:
		path (str): The folder where the dataset is located.
		shuffle (bool, optional): whether to use shuffling. Defaults: False.

	Returns:
		list: the list of file path.
	"""
	res = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			res.append(f)
	res.sort()	
	if shuffle:
		random.shuffle(res)
	return res
    

def generate_file_list(dataset_root, dataset, dataset_split, separator=" "):
	"""
	Create file list for the `HybridDataset`.

	Please make sure all datasets are placed in the same folder.

	Args:
		dataset_root (str): the root path of datasets.
		dataset (str): select datasets to join.
		dataset_split (str): split the train set or test set.
		separator (str, optional): The separator of train_file or val_file. If file name contains ' ', '|' may be perfect. Default: ' '.
	"""
	print("current dataset:", dataset)
	bg_path = DATASET_PATHS_DICT['BG20K']['TRAIN']['ORIGINAL_PATH']
	fg_path = DATASET_PATHS_DICT[dataset]['TRAIN']['FG_PATH']
	alpha_path = DATASET_PATHS_DICT[dataset]['TRAIN']['ALPHA_PATH']
	sample_interval = DATASET_PATHS_DICT[dataset]['TRAIN']['SAMPLE_INTERVAL']

	bg_list = get_files(bg_path, shuffle=True)
	alpha_list = get_files(alpha_path)

	file_list = os.path.join(dataset_root, dataset_split + ".txt")
	if os.path.exists(file_list):
		f = open(file_list, "a")
	else:
		f = open(file_list, "w")

	for idx, item in enumerate(alpha_list):
		for i in range(sample_interval):
			bg_index = idx * sample_interval + i
			file_path = os.path.join(fg_path, item)
			file_path = file_path + separator + os.path.join(bg_path, bg_list[bg_index])
			file_path += '\n'

			f.write(file_path)
			print(file_path)
	f.close()
    
if __name__ == '__main__':
	args = parse_args()
	for dataset in args.datasets:
		generate_file_list(args.dataset_root, dataset, args.split)