# SAR Ship Segmentation — DeepLabV3+

Semantic segmentation research project: DeepLabV3+ (custom ASPP + decoder, ImageNet-pretrained Xception encoder) applied to [HRSID](https://github.com/chaozhong2010/HRSID), a Synthetic Aperture Radar (SAR) ship segmentation dataset, in [`HRSID_Segmentation.ipynb`](HRSID_Segmentation.ipynb).

## Why SAR

SAR is a radar imaging modality, not optical: grayscale, speckle-noise-heavy, works day/night through cloud cover. Segmenting it is a genuinely different problem from RGB semantic segmentation — different edge statistics, different noise model.

## Dataset

**[HRSID](https://github.com/chaozhong2010/HRSID)** — 5,604 high-resolution SAR images (800×800), 16,951 ship instances, pixel-level instance masks, built from TerraSAR-X/Sentinel-1B/TanDEM-X scenes, COCO-format annotations.

## Architecture

- **Decoder / ASPP**: custom implementation (`SeparableConv`, `ASPP`, `DecoderBlock`), originally designed for a companion Cityscapes project.
- **Encoder**: a from-scratch `XceptionModel` is kept in the notebook for reference but **not used for training** — an ImageNet-pretrained Xception (`timm`) is used instead. Training a 16-block Xception from scratch needs far more epochs than a single Colab session allows; the pretrained backbone converges in a handful of epochs. See the markdown note in `HRSID_Segmentation.ipynb` for the full reasoning.

## Experiments

Run entirely in Google Colab (T4 GPU — see note below). Each has a "Write here" markdown cell for your own interpretation, not pre-filled analysis:

1. **Transfer learning** — frozen vs. fine-tuned pretrained encoder
2. **Grad-CAM** — where the model looks when predicting "ship"
3. **Failure case analysis** — high-confidence, low-IoU validation images
4. **Boundary IoU vs. interior IoU** — isolates DeepLabV3+'s known weak point (fine boundary precision) from overall accuracy
5. **Calibration / ECE** — is the model's confidence trustworthy?
6. **Accuracy-vs-latency** — cost of fine-tuning vs. frozen-encoder inference

## Results

| | |
|---|---|
| ![training curves](docs/training_curves.png) | ![grad-cam](docs/gradcam.png) |
| Frozen vs. fine-tuned loss curves | Grad-CAM overlay on a validation SAR image |
| ![failure cases](docs/failure_cases.png) | ![calibration](docs/calibration.png) |
| Confident-but-wrong failure cases | Reliability diagram (ECE) |

| Variant | Val mIoU | Full mIoU | Boundary mIoU | Boundary gap | Latency (ms/batch) |
|---|---|---|---|---|---|
| Frozen encoder | 0.766 | 0.656 | 0.657 | -0.001 | 267.6 |
| Fine-tuned | 0.764 | 0.656 | 0.642 | +0.013 | 267.5 |

The frozen encoder essentially ties (and marginally beats) full fine-tuning on val mIoU, with identical inference latency — and, counter-intuitively, fine-tuning introduces a small boundary-precision penalty that frozen doesn't have. See the notebook's "Final synthesis" section for the full interpretation.

*Val mIoU and Full mIoU are two different metrics, not a typo* — Val mIoU is `iou_score` from the training loop (batch-averaged, with an epsilon term); Full mIoU is `evaluate_boundary_vs_full`'s per-image IoU averaged over the full validation set (no epsilon, images with an empty union excluded). They measure the same thing but aren't numerically comparable to each other — only Val-to-Val and Full-to-Full comparisons across variants are meaningful.

## Running it

**Primary path — Google Colab** (this is where training actually happens):
1. Open `HRSID_Segmentation.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU** (not A100 — on Colab, A100 only gives its real speedup at half precision, which this project doesn't need; T4 is the better price/availability tradeoff here).
3. The setup cell downloads from a Kaggle mirror (`sarribere99/high-resolution-sar-images-dataset-hrsid`) — search Kaggle for "HRSID" yourself and swap the slug if it's moved.
4. Run top to bottom. Checkpoints and metrics are mirrored to Google Drive (`/content/drive/MyDrive/HRSID-experiment`) so a disconnected/restarted runtime resumes instead of retraining from scratch.

**Secondary path — local viewing via Docker** (browsing the notebook and its saved outputs, not training — there's no GPU passthrough configured):
```bash
make build
make jupyter
# open http://localhost:8888
```

**Linting** (for any future `.py` scripts — the notebook itself is excluded from Ruff/pre-commit, see `pyproject.toml`):
```bash
make setup   # installs deps + registers pre-commit hooks
make lint
make format
```

## Repo structure

```
HRSID_Segmentation.ipynb     # the project — SAR ship segmentation + experiments
requirements.txt             # dependency list (mirrors pyproject.toml)
pyproject.toml                # Ruff config, project metadata
.pre-commit-config.yaml       # pre-commit hooks (notebook excluded)
Dockerfile / Makefile         # local viewing environment
```

## Related

The ASPP/decoder architecture originates from a companion project, [CityScapesSegmentation](../CityScapesSegmentation) — a from-scratch DeepLabV3+ build on the Cityscapes street-scene dataset.

## License

MIT — see [LICENSE](LICENSE).
