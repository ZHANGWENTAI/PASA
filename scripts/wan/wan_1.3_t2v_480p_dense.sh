resolution="480p"
infer_step=50

first_times_fp=1
first_layers_fp=1

pattern="SAP"

output_dir="result/wan/t2v/dense"

# Video Cfg
video_cfg="Step_${infer_step}-Res_${resolution}"

# Dense Attention Cfg
dense_attention_cfg="TFP_${first_times_fp}-LFP_${first_layers_fp}"

# Output feature
output_feature="${video_cfg}/${dense_attention_cfg}"

for i in {1..10}; do
    prompt=$(cat examples/${i}/prompt.txt)
    export WAN_HIDDEN_L1_PROMPT_ID=${i}
    python wan_t2v_inference.py \
        --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
        --prompt "${prompt}" \
        --height 512 \
        --width 768 \
        --seed 0 \
        --num_inference_steps $infer_step \
        --pattern $pattern \
        --first_times_fp $first_times_fp \
        --first_layers_fp $first_layers_fp \
        --output_file "${output_dir}/${output_feature}/${i}-DENSE.mp4" \
        --logging_file "${output_dir}/${output_feature}/${i}-DENSE.jsonl"
done
