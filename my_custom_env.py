# my_envs/my_custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from quadcopters import Quadcopter
from rendering import QuadcopterRenderer

class MyCustomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=0.0,
            high=20.0,
            shape=(2,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.max_steps = 200
        self.current_step = 0

        self.quad = Quadcopter(0.8, 0.5, 1e-3)
        self.qr = QuadcopterRenderer(self.quad)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Randomize state upon reset
        self.quad.state = np.array([
            self.np_random.uniform(-0.01, 0.01), # y
            self.np_random.uniform(-0.01, 0.01), # z
            self.np_random.uniform(-0.01, 0.01), # phi
            self.np_random.uniform(-0.01, 0.01), # y_dot
            self.np_random.uniform(-0.01, 0.01), # z_dot
            self.np_random.uniform(-0.01, 0.01) # phi_dot
        ], dtype=np.float32)
        return self.quad.state, {}
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Simulate environment dynamics
        self.quad.step(0.01, action)
        
        self.current_step += 1
        
        # Calculate reward
        reward = float(-np.sum(
                (self.quad.state[3:5] ** 2) + (self.quad.state[0:2] ** 2)
            ))
        
        # Episode ends after max_steps or if state is large
        terminated = bool(np.sum(np.square(self.quad.state[:2])) > 1.0)
        truncated = self.current_step >= self.max_steps

        return self.quad.state, reward, terminated, truncated, {}

    def render(self):
        """Render the environment as an RGB array."""
        if self.render_mode != "rgb_array":
            return None
        
        # Get position and rotation from quad state
        position = np.append(0, self.quad.state[:2]) # (x, y, z)
        rotation = np.append(self.quad.state[2], [0, 0]) # (phi_x, phi_y, phi_z)

        return self.qr.render(position, rotation)

gym.register(
    id="MyCustomEnv-v0",
    entry_point=MyCustomEnv,
    max_episode_steps=500
)
