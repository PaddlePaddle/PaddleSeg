import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
def deal_annotation(mask_path,anno_path):
    for mask_name in tqdm(os.listdir(mask_path)):
        mask=Image.open(os.path.join(mask_path,mask_name))
        new_mask=mask.convert("L")
        # import pdb
        # pdb.set_trace()
        new_mask=np.asarray(new_mask).copy()
        new_mask.flags['WRITEABLE']=True
        new_mask[new_mask>0]=1
        new_mask=Image.fromarray(new_mask,"L")
        new_mask.save(os.path.join(anno_path,mask_name))
def data_download():
    pass
def unzipdataset():
    pass
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-data","--dataset_dir",default=r"D:\work\work001\Dataset\data")
    parser.add_argument("-split","--split_rate",default=0.15)
    args = parser.parse_args()
    dataset_path=args.dataset_dir
    val_rate=args.split_rate
    mask_path = os.path.join(dataset_path, "mask")
    anno_path=os.path.join(dataset_path,"annotation")
    os.makedirs(anno_path,exist_ok=True)
    deal_annotation(mask_path,anno_path)

    img_path=os.path.join(dataset_path,"npz")
    file_names=[file[:-4] for file in os.listdir(anno_path) if file.endswith(".png")]
    val_nums=int(val_rate*len(file_names))
    with open(os.path.join(dataset_path,"train.txt"),"w") as up:
        for filename in file_names[:-val_nums]:
            up.write(f"npz/{filename}.npy annotation/{filename}.png\n")

    with open(os.path.join(dataset_path,"val.txt"),"w") as up:
        for filename in file_names[-val_nums:]:
            up.write(f"npz/{filename}.npy annotation/{filename}.png\n")


if __name__=="__main__":
    main()