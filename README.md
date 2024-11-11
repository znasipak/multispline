# multispline
A Python package for generating cubic splines in multiple dimensions

# Installation

`multispline` relies on a few dependencies to install and run, namely
a C/C++ compiler (e.g., `g++`), `Cython`, `numpy`, and `python >= 3.7`, though we recommend using at least Python 3.9.
To reduce package conflicts and ensure that the proper dependencies are installed,
we recommend using Anaconda and its virtual environments.

Create a conda environment `spline-env` (or whatever name you would like)
with the necessary dependencies to install `multispline`. For MACOSX Intel run:
```
conda create -n spline-env -c conda-forge Cython numpy clang_osx-64 clangxx_osx-64 python=3.9
conda activate spline-env
```
This may also work for MACOSX silicon, though alternatively one should use:
For MACOSX Intel run:
```
conda create -n spline-env -c conda-forge Cython numpy clang_osx-arm64 clangxx_osx-arm64 python=3.9
conda activate spline-env
```
See Troubleshooting.
To instead include the necessary compiler on linux run:
```
conda create -n spline-env -c conda-forge Cython numpy gcc_linux-64 gxx_linux-64 python=3.9
conda activate spline-env
```
Next clone the :code:`multispline` repository from GitHub:
```
git clone https://github.com/znasipak/multspline.git
cd multispline
```
Finally, we recommend installing the package via `pip`:
```
pip install .
```

# Conda Environments with Jupyter

To run the code in a jupyter notebook, we recommend installing
the following dependencies through your preferred package manager,
such as `conda`: 
```
conda install ipykernel matplotlib
```
One can then make the environment accessible within Jupyter by
running
```
python -m ipykernel install --user --name=multispline-env
```
