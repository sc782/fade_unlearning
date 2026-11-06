CKPT_PATH=/data/UnlearnCanvas/ckpts/diffusion/compvis/style50/compvis.ckpt
DEVICE=4

for THEME in Monet Picasso Van_Gogh
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 stable_diffusion/train-scripts/train-esd-style.py \
  --theme ${THEME} \
  --train_method xattn \
  --ckpt_path ${CKPT_PATH}
done