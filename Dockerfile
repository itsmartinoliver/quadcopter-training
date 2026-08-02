FROM python:3.10

COPY pyproject.toml .
COPY ppo_continuous_action.py .
COPY my_custom_env.py .
COPY quadcopters.py .
COPY cleanrl_utils ./cleanrl_utils

RUN pip install .

ENTRYPOINT ["python", "ppo_continuous_action.py"]
CMD ["--env-id", "MyCustomEnv-v0", "--total-timesteps", "50000", "--seed", "1", "--save-model"]
