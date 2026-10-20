<div align="center">

# GEAM

### Graph-Enhanced Attribute-Aware Modeling for Cloth-Changing Person Re-Identification

Official PyTorch implementation

**Accepted by NeurIPS 2026**

![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-AE2D2D.svg)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
![Task CC-ReID](https://img.shields.io/badge/Task-CC--ReID-7B61FF.svg)

</div>

## 📊 Results

Rank-1 and mAP are reported in percent. SC, CC, and General denote the same-clothes, clothes-changing, and general evaluation protocols, respectively.

| Method | PRCC-SC R-1 | PRCC-SC mAP | PRCC-CC R-1 | PRCC-CC mAP | LTCC-General R-1 | LTCC-General mAP | LTCC-CC R-1 | LTCC-CC mAP |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| GEAM | **100.0** | **99.5** | **66.0** | **61.6** | **83.8** | **51.4** | **51.5** | **28.7** |

| Method | LaST R-1 | LaST mAP | Celeb-reID-light R-1 | Celeb-reID-light mAP | Celeb-reID R-1 | Celeb-reID mAP |
|:--|--:|--:|--:|--:|--:|--:|
| GEAM | **81.8** | **45.2** | **74.3** | **55.4** | **79.1** | **40.2** |

## 🛠️ Environment

The released environment was tested with:

- Python 3.9.21
- PyTorch 1.12.1
- torchvision 0.13.1
- timm 0.9.10
- CUDA-capable NVIDIA GPUs

Create an environment and install the dependencies:

```bash
conda create -n geam python=3.9.21 -y
conda activate geam
pip install -r requirements.txt
```

The first training run may download the pretrained EVA-02 backbone through `timm`. Evaluation from a GEAM checkpoint does not require downloading the backbone separately.

## 📁 Dataset Preparation

Download the five CC-ReID benchmarks from their official project pages:

- [PRCC](http://www.isee-ai.cn/~yangqize/clothing.html)
- [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
- [Celeb-reID and Celeb-reID-light](https://github.com/Huang-3/Celeb-reID)
- [LaST](https://github.com/shuxjweb/last)

GEAM uses offline 105-dimensional pedestrian attributes produced by a frozen SOLIDER attribute detector. The detector is not jointly optimized with GEAM. Precomputed attribute labels for all supported datasets are available on [Google Drive](https://drive.google.com/drive/folders/1WejujoMeDh2G0on14TNnk0LLj4GpdvV4?usp=sharing). Download the corresponding `PAR_PETA_105.txt` file and place it in the root of each dataset.

Alternatively, the attribute labels can be generated with the scripts from the [official SOLIDER repository](https://github.com/tinyvision/SOLIDER).

Each line of an attribute file follows:

```text
<image_path> <attribute_index> <0_or_1>
```

Organize the datasets as follows:

```text
DATA_ROOT/
├── prcc/
│   ├── rgb/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── PAR_PETA_105.txt
├── LTCC_ReID/
│   ├── train/
│   ├── query/
│   ├── test/
│   └── PAR_PETA_105.txt
├── Celeb-reID-light/
│   ├── train/
│   ├── query/
│   ├── gallery/
│   └── PAR_PETA_105.txt
├── Celeb-reID/
│   ├── train/
│   ├── query/
│   ├── gallery/
│   └── PAR_PETA_105.txt
└── last/
    ├── train/
    ├── val/
    │   ├── query/
    │   └── gallery/
    ├── test/
    │   ├── query/
    │   └── gallery/
    └── PAR_PETA_105.txt
```

The default dataset root is defined by `DATA.ROOT` in [config/defaults.py](config/defaults.py). You can either change it there or override it from the command line with `DATA.ROOT /path/to/datasets`.

> **Note:** The released evaluation configurations use `TEST.TYPE: image_only`. During evaluation, sample-specific attributes are replaced with an all-zero vector before the model forward pass. The current dataset loaders still read `PAR_PETA_105.txt`, so the attribute files must remain present when evaluating.

## 🚀 Training

The paper experiments use two NVIDIA RTX 3090 GPUs. The following example trains GEAM on LTCC with two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u distributed/launch.py --nproc_per_node=2 --master_port=6666 train.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True DATA.ROOT /path/to/datasets
```

The command above uses the local `distributed/launch.py` launcher. If the `distributed` directory is not included in the repository, use PyTorch's built-in launcher instead:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 train.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml MODEL.DIST_TRAIN True DATA.ROOT /path/to/datasets
```

Use the corresponding configuration for each dataset:

| Dataset | Configuration |
|:--|:--|
| PRCC | `configs/prcc/eva02_l_maskmeta_random.yml` |
| LTCC | `configs/ltcc/eva02_l_maskmeta_random.yml` |
| Celeb-reID-light | `configs/Celeb_light/eva02_l_maskmeta_random.yml` |
| Celeb-reID | `configs/Celeb/eva02_l_maskmeta_random.yml` |
| LaST | `configs/last/eva02_l_maskmeta_random.yml` |

Logs and the best checkpoint are written to the `OUTPUT_DIR` specified by the selected configuration. The complete commands for all datasets are also collected in [train_test.sh](train_test.sh).

## 📦 Pretrained Models

Pretrained GEAM checkpoints for the supported datasets are available on [Google Drive](https://drive.google.com/drive/folders/1LE_oxaE7-1iITq76gVQ3Nbg6my6sycJb?usp=sharing). Download the checkpoint corresponding to the target dataset and pass its path to `--weight` during evaluation.

## 🔍 Evaluation

### Recommended evaluation

`inference.py` performs one evaluation job, supports one or multiple GPUs, and reports Rank-1/mAP together with model complexity and runtime statistics:

```bash
python inference.py --config-file configs/ltcc/eva02_l_maskmeta_random.yml --weight /path/to/eva02_l_meta_best.pth --devices 0,1 --batch-size 128 DATA.ROOT /path/to/datasets
```

Useful options include:

- `--devices 0` or `--devices 0,1`: physical GPU IDs.
- `--batch-size 128`: global evaluation batch size.
- `--workers 4`: number of data-loading workers.
- `--amp`: enable automatic mixed precision.
- `--skip-flops`: skip the additional FLOPs-profiling forward pass.

For PRCC, the evaluator reports both clothes-changing and same-clothes results. For LTCC, it reports clothes-changing and general results. The other datasets use their default protocols.

### Legacy distributed entry point

The original evaluation entry point remains available. Using one process avoids duplicated evaluation output:

```bash
CUDA_VISIBLE_DEVICES=0 python -u distributed/launch.py --nproc_per_node=1 --master_port=6661 test.py --config_file configs/ltcc/eva02_l_maskmeta_random.yml TEST.WEIGHT /path/to/eva02_l_meta_best.pth DATA.ROOT /path/to/datasets
```

If the local `distributed` directory is unavailable, replace `python -u distributed/launch.py` with `python -u -m torch.distributed.launch`.

## 🙏 Acknowledgements

This repository builds upon excellent open-source projects including [MADE](https://github.com/moon-wh/MADE), [TransReID](https://github.com/damo-cv/TransReID), [CAL](https://github.com/guxinqian/Simple-CCReID), [EVA-02](https://github.com/baaivision/EVA), and [SOLIDER](https://github.com/tinyvision/SOLIDER). We sincerely thank their authors for making their work publicly available.

## 📖 Citation

If you find GEAM useful in your research, please cite:

```bibtex
@misc{ding2026geam,
  title  = {GEAM: Graph-Enhanced Attribute-Aware Modeling for Cloth-Changing Person Re-Identification},
  author = {Ding, Yongkang and Ye, Zi and Gong, Tiantian and Zhang, Liyan},
  year   = {2026}
}
```
