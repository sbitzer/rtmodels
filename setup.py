# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='rtmodels',
    version='0.1.0',
    author='Sebastian Bitzer',
    author_email='sebastian.bitzer@tu-dresden.de',
    packages=['rtmodels'],
    description='Models of response time.',
    zip_safe=False, # needed, because otherwise numba cannot cache
    install_requires=['NumPy >=1.7.0', 'matplotlib', 'seaborn', 'numba'],
    classifiers=[
                'Development Status :: 3 - Alpha',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Programming Language :: Python :: 3',
                'Topic :: Scientific/Engineering',
                 ]
)