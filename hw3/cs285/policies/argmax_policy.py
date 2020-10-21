import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        # actions is actually (batch_size=1, )
        if not isinstance(observation, torch.Tensor):
            observation = ptu.from_numpy(observation)

        actions = ptu.to_numpy(self.critic.q_net(observation).argmax(
            dim=-1,
            keepdim=False
        ))

        return actions.squeeze()