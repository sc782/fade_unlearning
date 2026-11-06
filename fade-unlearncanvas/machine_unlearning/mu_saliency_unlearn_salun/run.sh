DEVICE=5
OUTPUT_DIR=./results/style50
CKPT_PATH=/data/UnlearnCanvas/ckpts/diffusion/compvis/style50/compvis.ckpt

for THEME in Monet Picasso Van_Gogh
do
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 generate_mask.py --theme ${THEME} --output_dir ${OUTPUT_DIR} --ckpt_path ${CKPT_PATH}
  CUDA_VISIBLE_DEVICES=${DEVICE} python3 train-erase.py --theme ${THEME} --output_dir ${OUTPUT_DIR} --ckpt_path ${CKPT_PATH}
done