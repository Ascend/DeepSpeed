import setuptools

setuptools.setup(
    name="deepspeed_npu",
    version="0.1",
    description="An adaptor for deepspeed on Ascend NPU",
    packages=['deepspeed_npu'],
    install_package_data=True,
    include_package_data=True,
    install_requires=[
        "deepspeed==0.9.2"
    ],
    entry_points={
        "console_scripts": [
            "deepspeed=deepspeed_npu.cli:deepspeed_npu_main",
            "ds=deepspeed_npu.cli:deepspeed_npu_main",
        ],
    },
    license='Apache2',
    license_file='./LICENSE',
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
