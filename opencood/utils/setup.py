'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-10-31 16:54:56
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-11-17 23:59:03
'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    name='box overlaps',
    # ext_modules=cythonize('opencood/utils/box_overlaps.pyx'),
    ext_modules=cythonize('/home/jinlong/4.3D_detection/Noise_V2V/v2vreal/opencood/utils/box_overlaps.pyx'),
    include_dirs=[numpy.get_include()]
)