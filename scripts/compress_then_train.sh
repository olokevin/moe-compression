MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}" 
WANDB_PROJECT="${WANDB_PROJECT:-qwen2_5-0_5B-KD}"

TRAIN_MODE="${TRAIN_MODE:-svd_llm_v2}"  ### svd, svd_llm, svd_llm_v2, btt, btt_llm_v2, svd_llm_v2_bp, svd_llm_v2_combined, svd_als, svd_twosteps
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.8}"  ### 0.8 means 20% compression
KD_LOSS_TYPE="${KD_LOSS_TYPE:-ce}"  ### sft, ce, kl, kl_online
TEACHER_DATA_DIR="${TEACHER_DATA_DIR:-/data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math}"
LR="${LR:-1e-4}"
NAME_SUFFIX="${NAME_SUFFIX:-}"

if [[ "$KD_LOSS_TYPE" == "ce" ]]; then
    config_source=recipes/kd/config_ce.yaml
else
    config_source=recipes/kd/config_sft.yaml
fi

declare -ga cfg_suffix_args=()
if [[ -n "${CFG_SUFFIX:-}" ]]; then
  read -r -a cfg_suffix_args <<< "${CFG_SUFFIX}"
fi

declare -ga name_suffix_args=()
if [[ -n "${NAME_SUFFIX}" ]]; then
  name_suffix_args=(--name_suffix "${NAME_SUFFIX}")
fi

python src/compress_then_train.py --config $config_source \
    --model_name_or_path $MODEL_NAME \
    --wandb_project $WANDB_PROJECT \
    --train_mode $TRAIN_MODE \
    --compression_ratio $COMPRESSION_RATIO \
    --kd_loss_type $KD_LOSS_TYPE \
    --teacher_data_dir $TEACHER_DATA_DIR \
    --lr $LR \
    --save_steps none \
    "${name_suffix_args[@]}" \
    "${cfg_suffix_args[@]}"


# --save_steps none \
# --save_steps "0,10,final" \