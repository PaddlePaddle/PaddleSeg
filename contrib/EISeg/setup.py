import pathlib
from setuptools import setup, find_packages
from Cython.Build import cythonize
from eiseg import __APPNAME__, __VERSION__

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

setup(
    name=__APPNAME__,
    version=__VERSION__,
    description="交互式标注软件",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/PaddleSeg",
    author="PaddleSeg & PaddleCV-SIG",
    author_email="linhandev@qq.com",
    license="Apache Software License",  # 这里和readme的license不一样，统一了下，不知道是不是apache
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    # ext_modules=cythonize(["EISeg/cython_dict/_get_dist_maps.pyx"]),
    packages=find_packages(exclude=("test",)),
    # packages=["EISeg"],
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": [
            "eiseg=eiseg.run:main",
        ]
    },
)