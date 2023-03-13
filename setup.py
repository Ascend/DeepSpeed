import shutil
import subprocess
import os
import pkg_resources
import setuptools
from setuptools.command.install import install as _install


def _post_install():
    p = subprocess.Popen('which deepspeed', stdout=subprocess.PIPE, shell=True)
    dp_bin_path = p.communicate()[0].decode('utf-8').strip()
    if not os.path.isfile(dp_bin_path):
        raise RuntimeError('deepspeed executable file not found, installation will stop...')
    shutil.copyfile('bin/deepspeed', dp_bin_path)
    shutil.copyfile('bin/ds', os.path.join('/', *dp_bin_path.split('/')[:-1]), 'ds')

class PostInstall(_install):
    def run(self):
        _install.run(self)
        _post_install()

required_dp_ver = '0.6.0'
if pkg_resources.get_distribution("deepspeed").version != required_dp_ver:
    raise RuntimeError('deepspeed version should be {}, installation will stop...'.format(required_dp_ver))

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
    cmdclass={'install': PostInstall}
)