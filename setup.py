from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='load_experts_ext',
    ext_modules=[
        CUDAExtension('load_experts_ext', [
            'load_experts.cpp',
            'load_experts_cuda.cu',
        ],
        extra_compile_args={'nvcc': ['-arch=compute_86', '-code=sm_86']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })