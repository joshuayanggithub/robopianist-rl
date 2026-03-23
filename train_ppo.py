from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from train import Args as BaseArgs, get_env, prefix_dict
from gymnasium_wrapper import RoboPianistGymWrapper



@dataclass(frozen=True)
class PPOArgs:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 5_000_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    tqdm_bar: bool = False
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False

    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.8
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

class WandbLoggingCallback(BaseCallback):
    """Logs train episode return and PPO loss metrics to W&B."""

    def __init__(self, log_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_step(self) -> bool:
        step = self.num_timesteps

        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                wandb.log(
                    {
                        "train/episode_return": info["episode"]["r"],
                        "train/episode_length": info["episode"]["l"],
                    },
                    step=step,
                )

        if step % self.log_interval == 0:
            if len(self.model.logger.name_to_value) > 0:
                metrics = {
                    f"train/{k}": v
                    for k, v in self.model.logger.name_to_value.items()
                }
                wandb.log(metrics, step=step)

        return True 


class WandbEvalCallback(EvalCallback):
    """EvalCallback that also logs eval metrics and video to W&B."""

    def __init__(self, eval_env, eval_dm_env, eval_freq, n_eval_episodes, verbose=1):
        super().__init__(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose,
            render=False,
        )
        self._eval_dm_env = eval_dm_env  

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            step = self.num_timesteps
            try:
                log_dict = prefix_dict("eval", self._eval_dm_env.get_statistics())
                wandb.log(log_dict, step=step)
            except Exception:
                pass

            try:
                music_dict = prefix_dict(
                    "eval", self._eval_dm_env.get_musical_metrics()
                )
                wandb.log(music_dict, step=step)
            except Exception:
                pass

            try:
                video_path = self._eval_dm_env.latest_filename
                video = wandb.Video(str(video_path), fps=4, format="mp4")
                wandb.log({"video": video, "global_step": step}, step=step)
                video_path.unlink()
            except Exception:
                pass

        return result



def _make_base_args(args: PPOArgs, record_dir=None) -> BaseArgs:
    """Build a BaseArgs instance from PPOArgs so we can call get_env()."""
    return BaseArgs(
        root_dir=args.root_dir,
        seed=args.seed,
        environment_name=args.environment_name,
        n_steps_lookahead=args.n_steps_lookahead,
        trim_silence=args.trim_silence,
        gravity_compensation=args.gravity_compensation,
        reduced_action_space=args.reduced_action_space,
        control_timestep=args.control_timestep,
        stretch_factor=args.stretch_factor,
        shift_factor=args.shift_factor,
        wrong_press_termination=args.wrong_press_termination,
        disable_fingering_reward=args.disable_fingering_reward,
        disable_forearm_reward=args.disable_forearm_reward,
        disable_colorization=args.disable_colorization,
        disable_hand_collisions=args.disable_hand_collisions,
        primitive_fingertip_collisions=args.primitive_fingertip_collisions,
        frame_stack=args.frame_stack,
        clip=args.clip,
        record_every=args.record_every,
        record_resolution=args.record_resolution,
        camera_id=args.camera_id,
        action_reward_observation=args.action_reward_observation,
        record_dir=record_dir,
    )



def main(args: PPOArgs) -> None:
    run_name = args.name or f"PPO-{args.environment_name}-{args.seed}-{time.time()}"

    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )

    base_args_train = _make_base_args(args)
    base_args_eval = _make_base_args(args, record_dir=experiment_dir / "eval")

    dm_train = get_env(base_args_train)
    dm_eval = get_env(base_args_eval)

    train_env = Monitor(RoboPianistGymWrapper(dm_train))
    eval_gym_env = Monitor(RoboPianistGymWrapper(dm_eval))

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],  # match SAC hidden_dims
            activation_fn=__import__("torch.nn", fromlist=["GELU"]).GELU,
        ),
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(experiment_dir / "tb"),
    )

    train_cb = WandbLoggingCallback(log_interval=args.log_interval)
    eval_cb = WandbEvalCallback(
        eval_env=eval_gym_env,
        eval_dm_env=dm_eval,
        eval_freq=args.eval_interval,
        n_eval_episodes=args.eval_episodes,
    )

    print(f"Starting PPO training: {run_name}")
    print(f"  Total steps : {args.max_steps:,}")
    print(f"  Rollout len : {args.n_steps}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Epochs/upd  : {args.n_epochs}")
    print(f"  Gamma       : {args.gamma}")

    model.learn(
        total_timesteps=args.max_steps,
        callback=[train_cb, eval_cb],
        progress_bar=args.tqdm_bar,
    )

    save_path = experiment_dir / "ppo_final"
    model.save(str(save_path))
    print(f"Model saved to {save_path}.zip")

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(PPOArgs, description=__doc__))
