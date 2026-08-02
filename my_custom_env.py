# my_envs/my_custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import time
from quadcopters import Quadcopter
import render_utils

class MyCustomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.canvas_height = 800
        self.canvas_width = 800
        self.pixels_per_meter = 800  # Adjust based on your coordinate system
        self.quad_image = None  # Will be loaded on first render
        self.bg_image = None

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
        
        # Create a blank canvas
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        
        if self.quad_image is None: # Load the quadcopter image (assumes it's stored as self.quad_image)
            self.quad_image = render_utils._load_image('quadcopter.png', (60, 60))
        if self.bg_image is None: # Load the background image (assumes it's stored as self.bg_image)
            self.bg_image = render_utils._load_image('bg.png', (800, 800))
        
        # Get position and rotation from quad state
        position = self.quad.state[:2]  # (x, y)
        rotation = self.quad.state[2]   # angle in radians
        
        # Draw the background and quadcopter image
        canvas = render_utils._draw_rotated_image(canvas, self.bg_image)
        canvas = render_utils._draw_rotated_image(canvas, self.quad_image, position, rotation, self.pixels_per_meter)
        
        return canvas

gym.register(
    id="MyCustomEnv-v0",
    entry_point=MyCustomEnv,
    max_episode_steps=500
)
