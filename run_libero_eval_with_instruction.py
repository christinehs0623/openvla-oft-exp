"""
run_libero_eval_with_instruction.py

Modified version of run_libero_eval.py that accepts a custom language instruction
via command-line argument, overriding the task's default instruction.

Added arguments:
    --instruction_override   Custom instruction string to use for all tasks.
                             If empty, uses the task's original instruction.
    --task_ids               Comma-separated list of task IDs to run (e.g. "0,3,7").
                             If not set, runs all tasks in the suite.
    --results_json           Path to save per-task success rates as JSON.

Usage:
    # Run with original instructions
    python run_libero_eval_with_instruction.py \
        --pretrained_checkpoint <CHECKPOINT> \
        --task_suite_name libero_spatial \
        --instruction_override ""

    # Run with null instruction
    python run_libero_eval_with_instruction.py \
        --pretrained_checkpoint <CHECKPOINT> \
        --task_suite_name libero_spatial \
        --instruction_override "" \
        --results_json results_null.json

    # Run with a specific counterfactual instruction on specific tasks
    python run_libero_eval_with_instruction.py \
        --pretrained_checkpoint <CHECKPOINT> \
        --task_suite_name libero_spatial \
        --task_ids "0,3" \
        --instruction_override "pick up the cookies and place them on the plate" \
        --results_json results_cf.json
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LIBERO environment
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    # Instruction control
    instruction_override: str = "ORIGINAL"  # "ORIGINAL" = use task default; "" = null; anything else = use as-is
    task_ids: str = ""                       # Comma-separated task IDs to run, e.g. "0,3,7". Empty = run all.

    # Output
    results_json: str = ""                   # Path to save per-task results JSON. Empty = don't save.

    # Logging
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7

    # fmt: on


def get_instruction(task_description, instruction_override):
    """Return the instruction to use based on override setting."""
    if instruction_override == "ORIGINAL":
        return task_description
    else:
        # Could be empty string (null) or any custom string (counterfactual)
        return instruction_override


def log_message(message, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def initialize_model(cfg):
    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found!"
        cfg.unnorm_key = unnorm_key

    return model, action_head, proprio_projector, noisy_action_projector, processor


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(cfg, env, instruction, model, resize_size,
                processor, action_head, proprio_projector,
                noisy_action_projector, initial_state, log_file):
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                # print("Using instruction:", instruction)                    
            
                actions = get_action(
                    cfg, model, observation, instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images

def run_logit_lens_episode(cfg, env, instruction, model, resize_size,
                           processor, action_head, proprio_projector,
                           noisy_action_projector, initial_state, log_file,
                           logit_lens_layers=[0, 15, 31]):
    
    results = {}  # {layer_idx: {"success": bool, "images": []}}
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # 對每個 layer 各跑一次獨立 trajectory
    for layer_idx in logit_lens_layers + ["final"]:  # final = layer 31 的正常輸出
        env.reset()
        obs = env.set_init_state(initial_state)
        
        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        replay_images = []
        t = 0
        success = False
        
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)
            
            if len(action_queue) == 0:
                actions, logit_lens = get_action(
                    cfg, model, observation, instruction,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    logit_lens_layers=logit_lens_layers,
                )
                
                # 決定這個 trajectory 用哪一層的 action
                if layer_idx == "final":
                    chosen_actions = actions        
                else:
                    chosen_actions = logit_lens[layer_idx]  
                
                action_queue.extend(chosen_actions)
            
            action = action_queue.popleft()
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
        
        results[layer_idx] = {"success": success, "images": replay_images}
    
    return results

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(log_filepath, "w")
    logger.info(f"Logging to: {log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Determine which tasks to run
    if cfg.task_ids.strip():
        task_ids_to_run = [int(x.strip()) for x in cfg.task_ids.split(",")]
    else:
        task_ids_to_run = list(range(num_tasks))

    # Log instruction mode
    if cfg.instruction_override == "ORIGINAL":
        instruction_mode = "original"
    elif cfg.instruction_override == "":
        instruction_mode = "null"
    else:
        instruction_mode = f"override: \"{cfg.instruction_override}\""
    log_message(f"Task suite       : {cfg.task_suite_name}", log_file)
    log_message(f"Instruction mode : {instruction_mode}", log_file)
    log_message(f"Tasks to run     : {task_ids_to_run}", log_file)

    # Evaluation loop
    total_episodes, total_successes = 0, 0
    per_task_results = {}

    for task_id in tqdm.tqdm(task_ids_to_run):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        # print(env.sim.model.camera_names) # ('frontview', 'birdview', 'agentview', 'sideview', 'galleryview', 'robot0_robotview', 'robot0_eye_in_hand')

        instruction = get_instruction(task_description, cfg.instruction_override)

        log_message(f"\nTask {task_id}: {task_description}", log_file)
        log_message(f"  Using instruction: \"{instruction}\"", log_file)

        task_episodes, task_successes = 0, 0

        cfg.logit_lens_layers = [0, 15, 31]
        if cfg.instruction_override == "ORIGINAL":
            condition_label = "original"
        elif cfg.instruction_override == "None":
            condition_label = "null"
        else:
            condition_label = "counterfactual"

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), leave=False):
            if cfg.logit_lens_layers is not None:
                # Logit lens mode：對每個 layer 各跑一次 trajectory
                results = run_logit_lens_episode(
                    cfg, env, instruction, model, resize_size,
                    processor, action_head, proprio_projector,
                    noisy_action_projector, initial_states[episode_idx], log_file,
                    logit_lens_layers=cfg.logit_lens_layers,
                )
                # 用 final layer 的結果來算 success rate（跟原本一致）
                success = results["final"]["success"]
                
                # 存每個 layer 的 video
                for layer_idx, result in results.items():
                    save_rollout_video(
                        result["images"], total_episodes,
                        success=result["success"],
                        task_description=f"{condition_label}__{task_description}__layer{layer_idx}",
                        task_name=f"{cfg.task_suite_name}_{task_id}",
                        log_file=log_file,
                    )
            else:
                success, replay_images = run_episode(
                    cfg, env, instruction, model, resize_size,
                    processor, action_head, proprio_projector,
                    noisy_action_projector, initial_states[episode_idx], log_file,
                )

                # save_rollout_video(
                #     replay_images, total_episodes, success=done, task_description=f"{condition_label}__{task_description}", log_file=log_file
                # )

                save_rollout_video(
                    replay_images, total_episodes,
                    success=success,
                    task_description=f"{condition_label}__{task_description}",
                    task_name=f"{cfg.task_suite_name}_{task_id}",
                    log_file=log_file,
                )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1
            log_message(f"  Episode {episode_idx+1}: {'SUCCESS' if success else 'FAIL'}", log_file)

        task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        log_message(f"Task {task_id} success rate: {task_sr:.1%} ({task_successes}/{task_episodes})", log_file)

        per_task_results[str(task_id)] = {
            "task_description": task_description,
            "instruction_used": instruction,
            "episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_sr,
        }

        env.close()

    # Final summary
    final_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
    log_message(f"\n{'='*60}", log_file)
    log_message(f"Instruction mode : {instruction_mode}", log_file)
    log_message(f"Total episodes   : {total_episodes}", log_file)
    log_message(f"Total successes  : {total_successes}", log_file)
    log_message(f"Overall success rate: {final_sr:.1%}", log_file)
    log_message(f"{'='*60}", log_file)

    # Per-task summary table
    log_message("\nPer-task results:", log_file)
    log_message(f"{'Task ID':<10} {'Success Rate':<15} {'Instruction'}", log_file)
    for tid, r in per_task_results.items():
        log_message(f"{tid:<10} {r['success_rate']:<15.1%} {r['instruction_used']}", log_file)

    # Save results JSON
    if cfg.results_json:
        output = {
            "instruction_mode": instruction_mode,
            "instruction_override": cfg.instruction_override,
            "task_suite": cfg.task_suite_name,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "overall_success_rate": final_sr,
            "per_task": per_task_results,
        }
        with open(cfg.results_json, "w") as f:
            json.dump(output, f, indent=2)
        log_message(f"Results saved to: {cfg.results_json}", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_sr})
        wandb.save(log_filepath)

    log_file.close()


if __name__ == "__main__":
    eval_libero()