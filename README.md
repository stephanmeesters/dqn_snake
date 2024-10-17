# DQN Snake

A snake avoid-the-obstacles game implemented in [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

Implemented using DQN with a slightly modified CnnPolicy (see CNN.py).
Each snake has it's own aligned field of view and can take the action NONE, LEFT, RIGHT

Install requirements:
```
pip install -r requirements.txt
```

Run training:
```
python train.py
```

Play pretrained model:
```
python play.py
```

<img src="snake-animation.gif" />
