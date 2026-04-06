resolution="720p"
infer_step=50

output_dir="result/hyvideo/t2v/pasa"

first_times_fp=0.1
first_layers_fp=0.03

base_density=0.15
# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}"

for i in {1..10}; do
    prompt=$(cat examples/${i}/prompt.txt)
    python hyvideo_t2v_inference.py \
        --model_id "tencent/HunyuanVideo" \
        --seed 0 \
        --height 720 \
        --width 1280 \
        --prompt "${prompt}" \
        --num_inference_steps $infer_step \
        --first_times_fp $first_times_fp \
        --first_layers_fp $first_layers_fp \
        --resolution $resolution \
        --pattern "PASA" \
        --output_file "${output_dir}/${output_feature}/${i}-PASA.mp4" \
        --logging_file "${output_dir}/${output_feature}/${i}-PASA.jsonl"\
        --base_density $base_density \
        --use_dynamic \
        --use_group \
        --use_random
done
