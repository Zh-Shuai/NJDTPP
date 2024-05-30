# Neural Jump-Diffusion Temporal Point Processes
The implementation of our ICML-2024 paper ["Neural Jump-Diffusion Temporal Point Processes"]().

## Dataset
The real-world datasets are from ["EasyTPP"](https://github.com/ant-research/EasyTemporalPointProcess) and ["NHP"](https://github.com/hongyuanmei/neurawkes).

## Installation
1. Install the dependencies
```
conda env create -f environment.yml
```
2. Activate the conda environment
```
conda activate NJDTPP
```
3. Unzip the data
```
unzip data.zip
```

## Reproducing the results from the paper
Go to the source directory:
```
cd experiments
```

This directory contains all experiments on three synthetic and six real-world datasets, for example:

- Earthquake dataset
```
python earthquake.py
```

## Citation
If you find this code useful, please consider citing our paper. Thanks!

```
@inproceedings{zhang2024neural,
  title={Neural Jump-Diffusion Temporal Point Processes},
  author={Zhang, Shuai and Zhou, Chuan and Liu, Yang and Zhang, Peng and Lin, Xixun and Ma, Zhi-Ming},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Acknowledgements and References
Parts of this code are based on and/or copied from the code of ["NJSDE"](https://github.com/000Justin000/torchdiffeq/tree/jj585) and ["SAHP"](https://github.com/QiangAIResearcher/sahp_repo).