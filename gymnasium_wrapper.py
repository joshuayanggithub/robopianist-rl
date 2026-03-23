import gymnasium as gym
import numpy as np
import dm_env


class RoboPianistGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: dm_env.Environment):
        super().__init__()
        self._env = env

        obs_spec = env.observation_spec()
        act_spec = env.action_spec()

        obs_dim = int(obs_spec.shape[-1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=act_spec.minimum.astype(np.float32),
            high=act_spec.maximum.astype(np.float32),
            shape=act_spec.shape,
            dtype=np.float32,
        )

        self._current_timestep: dm_env.TimeStep | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        timestep = self._env.reset()
        self._current_timestep = timestep
        obs = timestep.observation.astype(np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        timestep = self._env.step(action)
        self._current_timestep = timestep

        obs = timestep.observation.astype(np.float32)
        reward = float(timestep.reward) if timestep.reward is not None else 0.0
        terminated = timestep.last()
        truncated = False  
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass  

    def close(self):
        pass


    def get_statistics(self) -> dict:
        """Delegates to the underlying EpisodeStatisticsWrapper."""
        return self._env.get_statistics()

    def get_musical_metrics(self) -> dict:
        """Delegates to the underlying MidiEvaluationWrapper."""
        return self._env.get_musical_metrics()

    @property
    def latest_filename(self):
        """Delegates to the underlying PianoSoundVideoWrapper."""
        return self._env.latest_filename

    @property
    def random_state(self):
        return self._env.random_state
