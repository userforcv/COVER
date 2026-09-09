# COVER

Code for **COVER: Learning Visible Evidence from CLIP-Guided Counterfactual Occlusion for Occluded Person Re-Identification**.

COVER connects CLIP-guided source-target selection, exact mask supervision for token reliability, and Counterfactual Visible-Part Aggregation (CVPA). CLIP scoring, synthetic occlusion, and masks are not used during inference.

## Installation

```bash
conda create -n cover python=3.8
conda activate cover
pip install -r requirements.txt
```

The bundled OpenAI CLIP loader downloads the ViT-B/16 weight on first use.

## Data

Set `DATA_ROOT` to a directory with this layout:

```text
data/
  dukemtmcreid/
    Occluded_Duke/
    DukeMTMC-reID/
  Market-1501-v15.09.15/
  Occluded_REID/
    occluded_body_images/
    whole_body_images/
```

## Training

Core COVER on Occluded-Duke:

```bash
DATA_ROOT=/path/to/data GPU_ID=0 bash scripts/train_cover.sh
```

For Market-1501 or DukeMTMC-reID, override the dataset and output directory:

```bash
DATA_ROOT=/path/to/data bash scripts/train_cover.sh \
  DATASETS.NAMES market1501 OUTPUT_DIR outputs/cover_market1501

DATA_ROOT=/path/to/data bash scripts/train_cover.sh \
  DATASETS.NAMES dukemtmc OUTPUT_DIR outputs/cover_dukemtmc
```

For the complementary SIE+OLP setting reported in the paper, append:

```text
MODEL.SIE_CAMERA True MODEL.SIE_COE 1.0 MODEL.STRIDE_SIZE "[12, 12]"
```

To reuse Stage 1, append:

```text
SOLVER.STAGE1.SKIP True SOLVER.STAGE1.WEIGHT /path/to/stage1_checkpoint.pth
```

## Evaluation

```bash
DATA_ROOT=/path/to/data bash scripts/evaluate_cover.sh \
  /path/to/COVER_epoch_80.pth
```

For cross-dataset evaluation on Occluded-ReID with a Market-1501 checkpoint:

```bash
DATA_ROOT=/path/to/data bash scripts/evaluate_cover.sh \
  /path/to/market_checkpoint.pth \
  DATASETS.NAMES occluded_reid OUTPUT_DIR outputs/cover_occluded_reid
```

No evaluation command enables re-ranking.

## Reported Core Result

| Dataset | mAP | Rank-1 |
|---|---:|---:|
| Occluded-Duke | 63.8 | 72.4 |
