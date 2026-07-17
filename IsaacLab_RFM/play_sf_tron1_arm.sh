#!/usr/bin/env bash
set -e

cd /home/phi5090ii/UMI-ON-TRON/umi-on-tron-lab-main/IsaacLab_RFM

LOAD_RUN="${LOAD_RUN:-2026-07-16_19-04-50}"
CHECKPOINT="${CHECKPOINT:-model_3400.pt}"
NUM_ENVS="${NUM_ENVS:-50}"

PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/rsl_rl:$PWD/source/ext_loco:$PYTHONPATH" \
/home/phi5090ii/UMI-ON-TRON/conda_envs/isaaclab_tron/bin/python \
scripts/rsl_rl/ios_play.py \
  --task Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-Play-v0 \
  --num_envs "$NUM_ENVS" \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT"
