# my_envs/my_custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from quadcopters import Quadcopter
from rendering import QuadcopterRenderer
from text_utils import *
from quat_utils import *

class MyCustomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.last_action = None
        self.action_space = spaces.Box(
            low=np.array([0.0, -0.1, -0.1, -0.1], dtype=np.float32), # Different lower and upper bounds for each component
            high=np.array([9.81+5.0, 0.1, 0.1, 0.1], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.max_steps = 200
        self.current_step = 0
        self.current_episode = -1

        self.quad = Quadcopter()
        self.qr = QuadcopterRenderer(self.quad)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        self.current_step = 0
        self.current_episode += 1
        # Randomize state upon reset
        self.quad.state = np.array([
            0, # x
            0, # y
            0, # z
            1, # q_a
            0, # q_b
            0, # q_c
            0, # q_d
            0, # v_x
            0, # v_y
            0, # v_z
            0, # w_x
            0, # w_y
            0, # w_z
        ], dtype=np.float32)
        return self.quad.state, {}
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Simulate environment dynamics
        self.quad.step(0.01, action)
        self.last_action = action
        
        self.current_step += 1
        
        # Calculate reward
        reward = float(-np.sum(self.quad.state[0:3] ** 2))
        
        # Episode ends after max_steps or if state is large
        terminated = bool(np.sum(np.square(self.quad.state[:3])) > 1.0)
        truncated = self.current_step >= self.max_steps

        return self.quad.state, reward, terminated, truncated, {}

    def render(self):
        """Render the environment as an RGB array."""
        if self.render_mode != "rgb_array":
            return None
        
        return self.qr.render(title=f"Episode {self.current_episode} step {self.current_step}/{self.max_steps}",
                              label=f"action: {vscw(self.last_action)}\n"+
                                    f"position: {vscw(self.quad.state[0:3])}\n"+
                                    f"euler: {vscw(quaternion_to_euler(self.quad.state[3:7]))}\n"+
                                    f"velocity: {vscw(self.quad.state[7:10])}\n"+
                                    f"ang vel: {vscw(self.quad.state[10:13])}"
                            )

gym.register(
    id="MyCustomEnv-v0",
    entry_point=MyCustomEnv,
    max_episode_steps=500
)
