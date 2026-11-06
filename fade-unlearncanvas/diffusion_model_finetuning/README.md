# diffusion_model_finetuning

Fine-tunes Stable Diffusion v1.5 on UnlearnCanvas and computes FADE.

## Setup

1. Download the [UnlearnCanvas dataset](https://github.com/OPTML-Group/UnlearnCanvas)
   into `data/unlearn_canvas/` (and a retain-only copy, excluding
   Monet/Picasso/Van_Gogh, into `data/unlearn_canvas_retain/` -- or generate it
   yourself from the full copy using `constants/const.py`'s `theme_available` list).
2. Download the Stable Diffusion v1.5 base checkpoint into
   `stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt`
   (see [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)).
3. `conda env create -f ../environment.yaml && conda activate unlearn_canvas`

## Getting checkpoints

Most users should use the published checkpoints rather than retraining. You don't
need to run this yourself in most cases -- `compute_fade_all.sh` calls
`scripts/download_checkpoints.py` before doing anything else, downloading from
HuggingFace on first use and skipping anything already present locally on every run
after that. Run it directly if you want
explicit control over when the download happens (e.g. to pre-fetch before going
offline):

```bash
./scripts/download_checkpoints.sh              # everything (~95GB)
./scripts/download_checkpoints.sh --only retain # just the 3 retain-only oracles
```

See the [checkpoint collection](https://huggingface.co/collections/sungjuncho/fade-unlearncanvas-checkpoints-6a61761b36b8f53ae21b57b8)
for what's available. To train your own instead (a locally-trained checkpoint at the
same path a downloaded one would occupy is used as-is -- nothing gets overwritten):

```bash
./scripts/train_full.sh                 # full (50-style) starting-point model
./scripts/train_retain.sh 0             # retain-only oracle, seed 0
./scripts/train_retain.sh 1
./scripts/train_retain.sh 2
```

To reproduce the 5 baseline unlearning methods (needed for the "unlearned"
checkpoints), see `../machine_unlearning/*/README.md` -- each is run against the full
checkpoint independently of this directory.

## Computing FADE

```bash
./scripts/compute_fade_all.sh [GPU ...]   # every model vs retain_seed0, 3 styles -> output_fade/ (auto-downloads)
python compute_fade.py --theme <style> --driver <name>=<ckpt> --others <name>=<ckpt> ...  # one direction
python aggregate.py                       # FADE table from output_fade/
```

`compute_fade.py` samples images from the `--driver` model (stochastic DDIM, eta 1, 100
steps, guidance scale 9, 512px; 20 object classes per seed, `--seeds` default 0-99) and,
at every sampling step, adds up the closed-form KL between the driver's step and each
`--others` model's step -- see "The FADE metric" in [`../README.md`](../README.md). For
each other model it saves `output_fade/<style>/<driver>_vs_<other>.pt` with a `diffs`
tensor (one value per image, in nats per latent dimension; multiply by `latent_dim` for
nats) and `per_step` (the per-step terms). The driver's images go to
`output_images/<driver>/<style>/` (`--image_dir ''` to skip).

One sampling chain serves all the models it is scored against, so `compute_fade_all.sh`
runs `retain_seed0` once against all 8 other models, then each of them once against
`retain_seed0`. The `retain_seed0` run holds 9 models on one GPU (about 57GB peak). On a smaller GPU, split
`--others` over several calls: the driver's chain depends only on the seed, so this
gives the same values.

Quick check that the pipeline runs (one seed, 20 images per direction):

```bash
python compute_fade.py --theme Monet --seeds 0 --driver retain0=logs/retain_seed0.ckpt --others retain1=logs/retain_seed1.ckpt
python compute_fade.py --theme Monet --seeds 0 --driver retain1=logs/retain_seed1.ckpt --others retain0=logs/retain_seed0.ckpt
python aggregate.py
```

## Secondary evaluation (classification accuracy / FID)

UnlearnCanvas's own style/class classifier accuracy and FID metrics (not FADE) are
also available, e.g. to sanity-check that an unlearning method actually removed the
target style, on the images `compute_fade.py` saves:

```bash
./scripts/evaluate_images.sh output_images/CA/Monet output_eval/CA/Monet
```

Requires the UnlearnCanvas classifier checkpoints (`ckpts/cls_model/style50.pth`,
`ckpts/cls_model/style50_cls.pth`) from the
[UnlearnCanvas repository](https://github.com/OPTML-Group/UnlearnCanvas).

## Other fine-tuning options (inherited from UnlearnCanvas, not FADE-specific)

This fork also carries UnlearnCanvas's diffuser-format training pipeline and
InstructPix2Pix image-editing fine-tuning, unused by FADE itself:

- Diffuser-format training: `accelerate launch diffuser_sd_training.py --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 --dataset_name=OPTML-Group/UnlearnCanvas ...` -- see the [UnlearnCanvas README](https://github.com/OPTML-Group/UnlearnCanvas) for full flag documentation.
- InstructPix2Pix fine-tuning: `python3 main.py --name ip2p_unlearn_canvas --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7`, sampled via `sampling/image_editing/sample_compvis_single.py`.

## Directory overview

| Path | Purpose |
|---|---|
| `compute_fade.py` | One FADE direction: samples from one checkpoint and bounds its KL to others |
| `aggregate.py` | Sums the two directions of each pair into the FADE table |
| `constants/const.py` | Style/class lists, including the retain-only (forget-excluded) style list |
| `sampling/` | UnlearnCanvas's image generation from compvis- and diffuser-format checkpoints |
| `evaluation/` | Classification accuracy + FID (secondary metrics) |
| `edm_sampler/` | Vendored k-diffusion samplers, used by `stable_diffusion/exp_qualitative_eval.py` and `sampling/image_editing/sample_single.py` (not by FADE) |
| `scripts/` | All entry points described above |
