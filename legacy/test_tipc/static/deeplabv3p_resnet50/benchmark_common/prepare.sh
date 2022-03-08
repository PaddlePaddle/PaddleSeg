# prepare
pip install -r requirements.txt
pip install argparse
# Download test dataset and save it to PaddleSeg/data
# It automatic downloads the pretrained models saved in ~/.paddleseg
data_item=${1:-'cityscapes'} 
case ${data_item} in
    cityscapes)     
        wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar --no-check-certificate \
        -O dataset/cityscapes.tar
        tar -xf dataset/cityscapes.tar -C dataset/
        echo "dataset prepared done"  ;;
    *) echo "choose data_item"; exit 1;
esac
