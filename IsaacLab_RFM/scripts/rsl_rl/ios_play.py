# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# from rsl_rl.rsl_rl.algorithms import PPO_TS, PPO

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--debug_arm", action="store_true", help="Print J1-J6 action, position, velocity and torque.")
parser.add_argument("--debug_arm_interval", type=int, default=25, help="Steps between arm debug prints.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many policy steps; 0 runs forever.")
parser.add_argument("--dump_observation", type=str, default=None, help="Write the initial 65-D policy observation to NPZ.")
parser.add_argument("--diagnostics_only", action="store_true", help="Dump environment metadata/observations without loading PPO.")
parser.add_argument("--target_x", type=float, default=None, help="Fixed EE target X relative to the robot base [m].")
parser.add_argument("--target_y", type=float, default=None, help="Fixed EE target Y relative to the robot base [m].")
parser.add_argument("--target_z", type=float, default=None, help="Fixed EE target Z in world coordinates [m].")
parser.add_argument("--target_roll", type=float, default=None, help="Fixed EE target roll [rad].")
parser.add_argument("--target_pitch", type=float, default=None, help="Fixed EE target pitch [rad].")
parser.add_argument("--target_yaw", type=float, default=None, help="Fixed EE target yaw [rad].")
# parser.add_argument("--use_teleop", type=str, default=False, help="Use Device for interacting with environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import time

import ext_loco.tasks  # noqa: F401
from ext_loco.utils import Logger

# from rsl_rl.runners import OneStageRunner, TwoStageRunner
from rsl_rl.runners import ImplicitOneStageRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from ext_loco.tasks.loco_manipulation.EE_pose.config.sf_tron1_arm.agents.implicit_one_stage_cfg import (
    ImplicitOneStageRunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

from ext_loco.utils import export_contactNet_as_onnx, export_gru_as_onnx, export_actor_as_onnx

# for onnx policy testing
# import onnxruntime as ort
# from rsl_rl.runners.implicit_os_runner import TestOnnxPolicyWrapper

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Allow fixed-command play targets to be supplied directly on the command
    # line.  A value is represented as (value, value), so no random sampling is
    # introduced.  Unspecified components retain the task configuration.
    target_values = {
        "pos_x": args_cli.target_x,
        "pos_y": args_cli.target_y,
        "pos_z": args_cli.target_z,
        "roll": args_cli.target_roll,
        "pitch": args_cli.target_pitch,
        "yaw": args_cli.target_yaw,
    }
    if any(value is not None for value in target_values.values()):
        command_cfg = env_cfg.commands.EE_pose
        if not hasattr(command_cfg, "ranges"):
            raise ValueError(f"Task {args_cli.task!r} does not expose a fixed EE target range.")
        for name, value in target_values.items():
            if value is not None:
                setattr(command_cfg.ranges, name, (value, value))
        configured_target = {
            name: getattr(command_cfg.ranges, name) for name in target_values
        }
        print(f"[INFO] Fixed EE target ranges: {configured_target}")

    agent_cfg: ImplicitOneStageRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    arm_action_ids = []
    arm_joint_ids = []
    arm_names = [f"J{i}" for i in range(1, 7)]
    action_offset = 0
    action_joint_names = []
    for term_name in env.unwrapped.action_manager.active_terms:
        term = env.unwrapped.action_manager.get_term(term_name)
        joint_names = list(getattr(term, "_joint_names", []))
        action_joint_names.extend(joint_names)
        for local_id, joint_name in enumerate(joint_names):
            if joint_name in arm_names:
                arm_action_ids.append(action_offset + local_id)
        action_offset += term.action_dim
    print(f"[diagnostic] action_joint_names={action_joint_names}")
    if args_cli.debug_arm:
        robot = env.unwrapped.scene["robot"]
        arm_joint_ids, arm_joint_names = robot.find_joints(arm_names, preserve_order=True)
        print(f"[debug_arm] action ids: {arm_action_ids}")
        print(f"[debug_arm] joint ids: {list(zip(arm_joint_names, arm_joint_ids))}")

    if args_cli.dump_observation or args_cli.diagnostics_only:
        import numpy as np

        initial_obs, initial_extras = env.get_observations()
        initial_contact_history = initial_extras["observations"]["contactNet"]
        if args_cli.dump_observation:
            dump_path = os.path.abspath(args_cli.dump_observation)
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            np.savez(
                dump_path,
                policy_observation=initial_obs[0].detach().cpu().numpy(),
                contact_history=initial_contact_history[0].detach().cpu().numpy(),
                critic_observation=initial_extras["observations"]["critic"][0].detach().cpu().numpy(),
                action_joint_names=np.asarray(action_joint_names),
            )
            print(f"[diagnostic] wrote initial observations to {dump_path}")
        if args_cli.diagnostics_only:
            env.close()
            return

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner = ImplicitOneStageRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    runner.load(resume_path)
    device = torch.device(env.unwrapped.device)

    runner.device = device  

    runner.ppo_alg.actor_critic.to(device)
    if hasattr(runner, "obs_normalizer") and runner.obs_normalizer is not None:
        runner.obs_normalizer.to(device)
    if hasattr(runner, "contactNet") and runner.contactNet is not None:
        runner.contactNet.to(device)
    if hasattr(runner, "contactNet_obs_normalizer") and runner.contactNet_obs_normalizer is not None:
        runner.contactNet_obs_normalizer.to(device)
    if hasattr(runner, "gru") and runner.gru is not None:
        runner.gru.to(device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # for onnx policy testing
    # export_model_dir = ("/home/xhw/IsaacLab.locomani/logs/rsl_rl/ImplicitOneStage/14-49-00/exported")
    # actor_critic_session = ort.InferenceSession(os.path.join(export_model_dir, "actor.onnx"))
    # gru_session = ort.InferenceSession(os.path.join(export_model_dir, "gru.onnx"))
    # tf_encoder_session = ort.InferenceSession(os.path.join(export_model_dir, "contactNet.onnx"))
    # test_onnx_policy = TestOnnxPolicyWrapper(
    #     actor_critic_session,
    #     gru_session,
    #     tf_encoder_session,
    #     1,
    #     env.num_obs,
    #     runner.gru_cfg["gru_latent_dim"],
    #     runner.ppo_alg_cfg["next_obs_latent_dim"],
    #     device=env.unwrapped.device,
    # )

    # policy = runner.get_inference_vanilla_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    export_actor_as_onnx(
        runner.ppo_alg.actor_critic, 
        runner.num_obs,
        normalizer=runner.obs_normalizer, 
        path=export_model_dir, 
        filename="actor.onnx"
    )
    export_contactNet_as_onnx(
        runner.num_contactNet_obs,
        contactNet=runner.contactNet,
        normalizer=runner.contactNet_obs_normalizer,
        path=export_model_dir,
        filename="contactNet.onnx",
    )
    export_gru_as_onnx(
        gru_wrapper=runner.gru,
        path=export_model_dir,
        filename="gru.onnx",
    )
    # export_cn_policy_as_onnx(
    #     runner.num_obs,
    #     agent_cfg.contactNet.output_dim - agent_cfg.contactNet.latent_dim - 4,
    #     runner.ppo_alg.actor_critic,
    #     normalizer=runner.obs_normalizer,
    #     path=export_model_dir,
    #     filename="cn_policy.onnx",
    # )

    stop_state_log = 500
    robot_index = 0
    joint_index = 0
    logger = Logger(env.unwrapped.step_dt)
    dof_pos = 0
    step_dt = env.unwrapped.step_dt  

    # reset environment
    obs, extras = env.get_observations()
    critic_obs = extras["observations"].get("critic", None)
    cn_obs_history = extras["observations"].get("contactNet", None)
    obs, critic_obs, cn_obs_history = (
        obs.to(runner.device),
        critic_obs.to(runner.device),
        cn_obs_history.to(runner.device),
    )
    
    # simulate environment
    i = 0
    while simulation_app.is_running():
        loop_start = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            action = policy(obs, cn_obs_history)
            if args_cli.debug_arm and i % max(args_cli.debug_arm_interval, 1) == 0:
                robot = env.unwrapped.scene["robot"]
                arm_action = action[0, arm_action_ids].detach().cpu().numpy()
                arm_pos = robot.data.joint_pos[0, arm_joint_ids].detach().cpu().numpy()
                arm_vel = robot.data.joint_vel[0, arm_joint_ids].detach().cpu().numpy()
                arm_torque = robot.data.applied_torque[0, arm_joint_ids].detach().cpu().numpy()
                print(
                    f"[debug_arm step={i}] "
                    f"action={arm_action.round(3).tolist()} "
                    f"pos={arm_pos.round(3).tolist()} "
                    f"vel={arm_vel.round(3).tolist()} "
                    f"torque={arm_torque.round(3).tolist()}"
                )
            
            # for onnx policy testing
            # action_1_onnx = test_onnx_policy(obs[0, :], cn_obs_history[0, :])
            
            # env stepping
            obs, _, dones, infos = env.step(action)
            cn_obs_history = infos["observations"]["contactNet"]
            runner.ppo_alg.gru.reset_hidden_states(dones)
            i += 1
            if args_cli.max_steps > 0 and i >= args_cli.max_steps:
                break
        # 按真实时间节流到物理步长
        elapsed = time.time() - loop_start
        sleep_time = step_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
