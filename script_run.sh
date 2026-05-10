# bin/bash
export CUDA_VISIBLE_DEVICES=0
python run_all_conditions.py \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
    --task_suite_name libero_object \
    --conditions null \
    --counterfactual_map counterfactual_map.json \
    --output_dir ./results \
    --num_trials_per_task 20

python run_all_conditions.py \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --conditions null  \
    --counterfactual_map counterfactual_map.json \
    --output_dir ./results \
    --num_trials_per_task 20

python run_all_conditions.py \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --conditions null \
    --counterfactual_map counterfactual_map.json \
    --output_dir ./results \
    --num_trials_per_task 20

python run_all_conditions.py \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --conditions null \
    --counterfactual_map counterfactual_map.json \
    --output_dir ./results \
    --num_trials_per_task 20