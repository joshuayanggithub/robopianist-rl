#!/bin/bash
# PPO entropy/n_steps sweep for comparing against SAC baseline.
# Runs 3 configurations in sequence to test the entropy hypothesis.

BASE_FLAGS="
    --root-dir /tmp/robopianist/rl/ \
    --max-steps 5000000 \
    --gamma 0.8 \
    --batch-size 256 \
    --n-epochs 10 \
    --learning-rate 3e-4 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name RoboPianist-debug-TwinkleTwinkleRousseau-v0 \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id piano/back \
    --tqdm-bar \
    --mode online
"

export WANDB_DIR=/tmp/robopianist/
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

echo "=========================================="
echo "Run 1/3: ent_coef=0.01 (entropy hypothesis test)"
echo "=========================================="
python train_ppo.py \
    $BASE_FLAGS \
    --n-steps 2048 \
    --ent-coef 0.01 \
    --name "PPO-ent0.01-nsteps2048"

echo "=========================================="
echo "Run 2/3: ent_coef=0.05 (stronger entropy)"
echo "=========================================="
python train_ppo.py \
    $BASE_FLAGS \
    --n-steps 2048 \
    --ent-coef 0.05 \
    --name "PPO-ent0.05-nsteps2048"

echo "=========================================="
echo "Run 3/3: ent_coef=0.01 + n_steps=4096 (combined fix)"
echo "=========================================="
python train_ppo.py \
    $BASE_FLAGS \
    --n-steps 4096 \
    --ent-coef 0.01 \
    --name "PPO-ent0.01-nsteps4096"

echo "=========================================="
echo "All runs complete!"
echo "=========================================="