#!/usr/bin/env bash
set -e

cd /home/phi5090ii/UMI-ON-TRON/umi-on-tron-lab-main/IsaacLab_RFM

PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/rsl_rl:$PWD/source/ext_loco:$PYTHONPATH" \
/home/phi5090ii/UMI-ON-TRON/conda_envs/isaaclab_tron/bin/python \
scripts/rsl_rl/ios_train.py \
  --task Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0 \
  --num_envs 8192 \
  --headless \
  --logger wandb \
  --resume True \
  --load_run 2026-07-15_21-17-07 \
  --checkpoint model_15400.pt \
  --max_iterations 20000
