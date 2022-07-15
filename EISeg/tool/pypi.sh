rm dist/*
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/  dist/* --verbose
# https://upload.pypi.org/legacy/

conda create -n test python=3.9
conda activate test
pip install --upgrade eiseg
pip install paddlepaddle
eiseg
