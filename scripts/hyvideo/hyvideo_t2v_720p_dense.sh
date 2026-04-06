resolution="720p"
infer_step=50

output_dir="result/hyvideo/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

output_feature="${video_cfg}"

for i in {1..5}; do
    export HYVIDEO_NOISEPRED_L1_LOG="${output_dir}/${output_feature}/${i}-DENSE_calibration.jsonl"
    export HYVIDEO_NOISEPRED_L1_PROMPT_ID=${i}
    prompt=$(cat calibration_dataset/${i}/prompt.txt)
    python hyvideo_t2v_inference.py \
        --model_id "tencent/HunyuanVideo" \
        --seed 0 \
        --height 720 \
        --width 1280 \
        --prompt "${prompt}" \
        --num_inference_steps $infer_step \
        --resolution $resolution \
        --pattern "dense" \
        --output_file "${output_dir}/${output_feature}/${i}-DENSE_calibration.mp4" \
        --logging_file HYVIDEO_NOISEPRED_L1_LOG
done
