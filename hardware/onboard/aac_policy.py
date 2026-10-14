"""
Asymmetric actor-critic for SB3's PPO (Pinto et al. 2018, arXiv:1710.06542).

The observation is [actor inputs (actor_dim), privileged inputs]. The critic reads all of it; the
actor's features extractor multiplies the privileged columns by zero. Zero inputs give zero
gradients, so the actor's first-layer weights on those columns never move from where they start
(zero after vec_util.widen_policy_input, and irrelevant anyway): the actor's output is a function
of the first actor_dim inputs only. That is what lets the real drone, which has no privileged
state, pad those columns with zeros (policy.load_policy) and fly exactly what was trained.

Kept to one subclass of ActorCriticPolicy with share_features_extractor=False, so everything else
(PPO.load / save, predict, the BC term in train_onboard.BCRegularizedPPO) works unchanged.
"""
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MaskExtractor(BaseFeaturesExtractor):
    """Flatten, then zero every input from `actor_dim` on (same width out, no parameters)."""

    def __init__(self, observation_space: spaces.Box, actor_dim: int):
        width = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=width)
        mask = torch.zeros(width)
        mask[:int(actor_dim)] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.flatten(observations, start_dim=1) * self.mask


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """policy_kwargs: actor_dim (required) + the usual ActorCriticPolicy ones (net_arch, ...)."""

    def __init__(self, *args, actor_dim: int, **kwargs):
        object.__setattr__(self, "actor_dim", int(actor_dim))
        kwargs["share_features_extractor"] = False
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # replaced before the optimizer is made (it has no parameters, but its buffer moves with .to())
        self.pi_features_extractor = MaskExtractor(self.observation_space, self.actor_dim)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        return data
