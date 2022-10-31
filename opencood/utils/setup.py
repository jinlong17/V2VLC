'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-10-31 16:54:56
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-10-31 17:25:22
'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    name='box overlaps',
    ext_modules=cythonize('opencood/utils/box_overlaps.pyx'),
    # ext_modules=cythonize('/home/jinlong/4.3D_detection/Detection_CVPR/v2vreal/opencood/utils/box_overlaps.pyx'),
    include_dirs=[numpy.get_include()]
)