import setuptools
import pathlib
setuptools.setup(
    name="deepspeed_npu",
    version="0.1",
    description="An adaptor for deepspeed on Ascend NPU",
    packages=['deepspeed_npu'],
    install_package_data=True,
    include_package_data=True,
    license='Apache2',
    license_file='./LICENSE',
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",        
    ],
    python_requires=">=3.7",
)