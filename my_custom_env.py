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

        self.last_action = None
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
        self.current_episode = -1

        self.quad = Quadcopter(0.8, 0.5, 1e-3)
        self.qr = QuadcopterRenderer(self.quad)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_episode += 1
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
        self.last_action = action
        
        self.current_step += 1
        
        # Calculate reward
        reward = float(-np.sum(self.quad.state[0:2] ** 2))
        
        # Episode ends after max_steps or if state is large
        terminated = bool(np.sum(np.square(self.quad.state[:2])) > 1.0)
        truncated = self.current_step >= self.max_steps

        return self.quad.state, reward, terminated, truncated, {}

    def render(self):
        """Render the environment as an RGB array."""
        if self.render_mode != "rgb_array":
            return None
        
        return self.qr.render(title=f"Episode {self.current_episode} step {self.current_step}/{self.max_steps}",
                              label=f"action: {self.last_action}\n"+
                                    f"position: ({str(self.quad.state[0])[:6]}, {str(self.quad.state[1])[:6]})\n"+
                                    f"rotation: {str(self.quad.state[2])[:6]} rads\n"+
                                    f"velocity: ({str(self.quad.state[3])[:6]}, {str(self.quad.state[4])[:6]})\n"+
                                    f"angular velocity: {str(self.quad.state[5])[:6]}"
                            )

gym.register(
    id="MyCustomEnv-v0",
    entry_point=MyCustomEnv,
    max_episode_steps=500
)
