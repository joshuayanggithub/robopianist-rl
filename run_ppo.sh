#!/bin/bash
# Run PPO on Twinkle Twinkle Little Star.
# Uses the same environment flags as run.sh for a fair comparison with SAC.

WANDB_DIR=/tmp/robopianist/ \
MUJOCO_GL=egl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
python train_ppo.py \
    --root-dir /tmp/robopianist/rl/ \
    --max-steps 5000000 \
    --gamma 0.8 \
    --n-steps 2048 \
    --batch-size 256 \
    --n-epochs 10 \
    --learning-rate 3e-4 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar \
    --mode online
