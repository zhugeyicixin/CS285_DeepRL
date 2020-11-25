import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from typing import Union
from typing import Any

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.utils import normalize


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(
                self.logits_na.parameters(),
                self.learning_rate
            )
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            # TODO: shouldn't logstd also be a NN?
            self.logstd = nn.Parameter(torch.zeros(
                self.ac_dim, dtype=torch.float32, device=ptu.device
            ))
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
            self.normal_dist = distributions.Normal(
                ptu.from_numpy(0.0),
                ptu.from_numpy(1.0)
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from Piazza

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        action = ptu.to_numpy(self._get_action(observation))

        return action

    def _get_action(self, obs: np.ndarray) -> torch.Tensor:

        acs = self.forward(obs)
        if self.discrete:
            # the out in gym descrete is an index of action
            sampler = distributions.Categorical(logits=acs)
            acs = sampler.sample()
        else:
            # sample a N(0, 1) normal distribution
            epsilon = self.normal_dist.sample(acs.shape)
            acs = acs + epsilon * torch.exp(self.logstd)
        return acs


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(
        self,
        observation: Union[np.ndarray, torch.Tensor]
    ) -> Any:

        if not isinstance(observation, torch.Tensor):
            observation = ptu.from_numpy(observation)

        if self.discrete:
            # output logits
            output = self.logits_na(observation)
        else:
            # output mean
            output = self.mean_net(observation)

        return output


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    # MJ: cut acs_labels_na and qvals from the signature if they are not used
    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################
