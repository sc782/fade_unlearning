CKPT_PATH=/data/UnlearnCanvas/ckpts/diffusion/compvis/style50/compvis.ckpt
DEVICE=2

for THEME in Monet Picasso Van_Gogh
do
    CUDA_VISIBLE_DEVICES=${DEVICE} python3 train-erase.py \
    --theme ${THEME} \
    --ckpt_path ${CKPT_PATH}
done