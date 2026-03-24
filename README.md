# gsplat submodule for GSDD
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
[![arXiv](https://img.shields.io/badge/BeyondPixels-2509.26219-b31b1b)](https://arxiv.org/abs/2509.26219)

This is the gsplat module for the paper titled "GaussianImaParameterizing Dataset Distillation via Gaussian Splatting". This module supports rendering and optimizing multiple images in one single forward and backward process.

[[code](https://github.com/j-cyoung/GSDatasetDistillation)][[paper](https://arxiv.org/abs/2509.26219)]

## Installation

```bash
bash build_fash.sh
```

```bash
pip install -e .[dev]
```

For more detail development instructions, please refer to *./docs/DEV.md*.

## Acknowledgments

Our code was developed based on [gsplat](https://github.com/nerfstudio-project/gsplat) and [GaussianImage](https://github.com/Xinjie-Q/GaussianImage). This is a concise and easily extensible Gaussian splatting library.

## Citation

If you find our GaussianImage paradigm useful or relevant to your research, please kindly cite our paper:

```
@misc{jiang2026parameterizingdatasetdistillationgaussian,
      title={Parameterizing Dataset Distillation via Gaussian Splatting}, 
      author={Chenyang Jiang and Zhengcen Li and Hang Zhao and Qiben Shan and Shaocong Wu and Jingyong Su},
      year={2026},
      eprint={2509.26219},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.26219}, 
}
```
