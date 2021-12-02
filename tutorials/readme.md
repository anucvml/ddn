# DDN Tutorial

The JuPyter notebook tutorial requires several Python packages (see requirements.txt and requirements_forge.txt). We recommend using a conda environment if you don't want to install these packages natively.

1. Install Miniconda by following the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
1. Update conda
```
conda update conda
```
1. Create the conda environment using
```
conda create --name ddn_tutorial python=3.8
```
1. Activate the environment using
```
conda activate ddn_tutorial
```
1. Install dependencies using
```
conda install -c pytorch --file requirements.txt
```
and
```
conda install -c conda-forge --file requirements_forge.txt
```
If using Windows, you may also need to install pywin32
```
conda install pywin32
```
and possibly the IPython kernel
```
python -m ipykernel install --user
```
1. Start up the JuPyter notebook using
```
jupyter notebook <notebook path>/08_ddn_pytorch_node.ipynb
```
1. When you're done, close the notebook and deactivate the environment
```
conda deactivate
```

