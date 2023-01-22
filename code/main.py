import gymnasium as gym
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default=None, type=str, help="Model to select between 'dqn' and 'dueling'")
    parser.add_argument('--load_checkpoint', default=False, type=bool, help='Whether to load a checkpoint of the agent or train it')
    parser.add_argument('--filename', default=None, type=str, help='Name of the checkpoint of the model to load')
    parser.add_argument('--render', type=bool, default=False, help='Render the enviroment')


    args = parser.parse_args()

    model = args.model
    load_checkpoint = args.load_checkpoint
    filename = args.filename
    render = args.render

    num_episodes = 1000
    max_time_steps = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")

    agent = Agent(
        model=model, 
        state_space=8, 
        action_space=4,
        buffer_size=int(1e5),
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        tau=1e-3, 
        device=device
    )

    if load_checkpoint:
        print('Loading Model')
        agent.load_model(filename)
        agent.eps = 0.01

    scores = []
    score_window = deque(maxlen=100)
    mean_score = []

    for episode in range(1, num_episodes+1):

        state, _ = env.reset()
        score = 0
        step = 5

        for _ in range(max_time_steps):

            action = agent.action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Either truncated or terminated we set 'done' to True
            if truncated == True:
                terminated = True

            if not load_checkpoint:
                agent.buffer.add_element(state, action, reward, next_state, terminated)

            state = next_state
            score += reward
            step += 1 

            # Training every k steps to emulate the 'frame skipping' technique
            if (not load_checkpoint) and (step % 5 == 0):
                agent.train()
                agent.soft_copy()

            if truncated or terminated:
                break
        
        
        if not load_checkpoint:
            agent.update_eps()
            

            if episode % 100 == 0:
                filename = 'dqn_' + str(episode)
                agent.save_model(filename)

        score_window.append(score)
        scores.append(score)
        mean_score.append(np.mean(score_window))

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score_window)), end="")

        if episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score_window)))


    # Plot the mean score over 100 episodes
    # if agent.model == 'dqn':
    #     np.savetxt('dqn_score', mean_score)
    # else:
    #     np.savetxt('dueling_score', mean_score)

    plt.plot(range(len(mean_score)), mean_score)
    plt.show()