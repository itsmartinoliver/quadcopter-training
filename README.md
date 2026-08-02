# Quadcopter Training
Run `python ppo_continuous_action.py --env-id MyCustomEnv-v0 --total-timesteps 50000 --seed 1 --save-model --capture-video`
and `tensorboard --logdir runs`
to train and view results.

Custom environments must be imported in the cleanRL scripts to work
