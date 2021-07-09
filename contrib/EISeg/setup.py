import pathlib
from setuptools import setup, find_packages
from Cython.Build import cythonize

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text(encoding="utf-8")


setup(
    name="EISeg",
    version="0.1.6",
    description="交互式标注软件",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/PaddleCV-SIG/EISeg",
    author="Paddlecv-SIG",
    author_email="linhandev@qq.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    # ext_modules=cythonize(["EISeg/cython_dict/_get_dist_maps.pyx"]),
    packages=find_packages(exclude=("test",)),
    # packages=["EISeg"],
    include_package_data=True,
    install_requires=[
        "pyqt5",
        "qtpy",
        "opencv-python",
        "scipy",
        "paddleseg",
        "albumentations",
        "cython",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "eiseg=eiseg.run:main",
        ]
    },
)
