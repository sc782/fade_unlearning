CKPT_PATH=/data/UnlearnCanvas/ckpts/diffusion/compvis/style50/compvis.ckpt
THEMES=( Monet Picasso Van_Gogh )
BIGNAMES=( "Monet" "Picasso" "Van Gogh")
SMALLNAMES=( monet picasso van_gogh )
DEVICE=0

for t_idx in "${!THEMES[@]}"
do
    THEME=${THEMES[$t_idx]}
    BIGNAME=${BIGNAMES[$t_idx]}
    SMALLNAME=${SMALLNAMES[$t_idx]}

    CUDA_VISIBLE_DEVICES=${DEVICE} python train.py -t --gpus 0, \
    --concept_type style \
    --caption_target "${BIGNAME} Style" \
    --prompts ./anchor_prompts/finetune_prompts/painting.txt \
    --name "ca_${SMALLNAME}"  \
    --train_size 200 \
    --resume-from-checkpoint-custom ${CKPT_PATH}
done