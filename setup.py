from setuptools import setup, find_packages

contrib = [
    'Raphael Ortiz',
    'Markus Rempfler',
]

setup(
    name='RDCNet',
    version='0.1',
    description=
    'RDCNet: Instance segmentation with a minimalist recurrent residual network (MICCAI-MLMI 2020)',
    author=', '.join(contrib),
    license='MIT',
    packages=find_packages(exclude=[
        'tests',
    ]),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'tensorflow>=2.3',
        'scikit-image>=0.13',
        'scikit-learn',
        'future',
        'pytest',
        'pandas',
    ],
    zip_safe=False)
