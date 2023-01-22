Basic usage.

These commands will start a training session over 1000 episodes. The checkpoints will override the existing ones in the folder 'checkpoints'.

>> python main.py --model=dqn
>> python main.py --model=dueling


To load a pre-trained model and render the environment run the following command

>> python main.py --model=dueling --load_checkpoint=True --filename=dueling_1000 --render=True
