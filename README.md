# README

My learning project based on the book https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

I added some of the improvements (change ReLu to Mish) and other class organization

To start training, enter the command:
`python main.py --cuda`

To start monitoring via tensorboardX:
tensorboard --logdir runs
`runs - folder in root of project`

To play back the results of the model:
`python dqn_play.py -m PongNoFrameskip-v4-best.dat`