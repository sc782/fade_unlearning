# Usage

This method is based on the environment `ablate`, see `environment.yaml`.


The checkpoint used for CA is in the compvis format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) `compvis/style50/compvis.ckpt`. One example is as follows:

```bash
python train.py -t --gpus 0, --concept_type style --caption_target  "Van Gogh Style" --prompts ./anchor_prompts/finetune_prompts/painting.txt --name "ca_van_gogh"  --train_size 200 --resume_from_checkpoint_custom PATH_TO_COMPVIS_CHECKPOINT
```

Please see `style_script.py` and `class_script.py` for runs for unlearning each style and object class.

## Pre-trained checkpoint

The resulting checkpoints for Monet/Picasso/Van_Gogh are published at
[sungjuncho/fade-unlearncanvas-unlearned-ca](https://huggingface.co/sungjuncho/fade-unlearncanvas-unlearned-ca) --
download via `../../diffusion_model_finetuning/scripts/download_checkpoints.sh` instead
of re-running the above if you just need the checkpoints. Note: use `last-v1.ckpt`
under `logs/ca_<style>/checkpoints/`, not `last.ckpt` -- the latter is a stale
checkpoint left over from an interrupted training run (epoch=1/step=1), not the fully
trained one (epoch=19/step=1000).