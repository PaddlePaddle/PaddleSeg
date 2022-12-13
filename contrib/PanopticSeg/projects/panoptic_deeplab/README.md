CITYSCAPES_DIR=''
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
pip install pyquaternion
export PYTHONPATH="$(pwd)"
python cityscapesscripts/preparation/createPanopticImgs.py --dataset-folder "${CITYSCAPES_DIR}/gtFine" --use-train-id --set-names train
python cityscapesscripts/preparation/createPanopticImgs.py --dataset-folder "${CITYSCAPES_DIR}/gtFine" --set-name val test
