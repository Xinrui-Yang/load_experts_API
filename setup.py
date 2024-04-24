from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='load_experts_ext',
    ext_modules=[
        CUDAExtension('load_experts_ext', [
            'load_experts.cpp',
            'load_experts_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })