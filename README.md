# Neural Jump-Diffusion Temporal Point Processes
The implementation of our ICML-2024 (Spotlight) paper ["Neural Jump-Diffusion Temporal Point Processes"](https://openreview.net/forum?id=d1P6GtRzuV).

## Updates
**Updates** (July 18, 2025)

**1. Improved Training Speed:** The training process now performs computations only on valid positions within padded sequences, eliminating unnecessary calculations for padding tokens. This optimization effectively reduces computational overhead, especially for datasets with large variations in sequence length.

**2. Improved Inference Speed:** Inference has been optimized to handle batch predictions at once, rather than processing one sequence at a time, resulting in significantly faster inference times.

**3. Upper Limit Estimation for Integral:** The upper limit ($\infty$) of the integral in Eq.(21) of our paper has been truncated using training data instead of test data. This change ensures a more reasonable approach.

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

- MIMIC-II dataset
```
python mimic2.py
```

## Citation
If you find this code useful, please consider citing our paper:

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