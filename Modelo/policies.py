"""
policies.py
-----------
Política actor-crítico asimétrica para PPO (Stable Baselines3).

El actor (la red que se lleva al robot) solo ve las primeras `actor_dim`
componentes de la observación: lo que darán la cámara y el firmware. El crítico
(solo existe durante el entrenamiento) ve además el estado real del simulador,
lo que le permite estimar mucho mejor el valor de cada situación aunque la
información del actor sea ruidosa, tardía o incompleta. Es la técnica estándar
de entrenamiento asimétrico para transferir de simulación a robot real
(Pinto et al., 2018).

`actor_weights(model)` extrae los pesos del actor como arrays de NumPy para
portarlos al PC de control o al STM32.
"""

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


def _mlp(in_dim, sizes, activation_fn):
    layers, last = [], in_dim
    for size in sizes:
        layers += [nn.Linear(last, size), activation_fn()]
        last = size
    return nn.Sequential(*layers), last


class AsymmetricExtractor(nn.Module):
    def __init__(self, feature_dim: int, actor_dim: int, net_arch, activation_fn):
        super().__init__()
        if isinstance(net_arch, dict):
            pi, vf = net_arch.get("pi", [256, 256]), net_arch.get("vf", [256, 256])
        else:
            pi = vf = list(net_arch)
        self.actor_dim = actor_dim
        self.policy_net, self.latent_dim_pi = _mlp(actor_dim, pi, activation_fn)
        self.value_net, self.latent_dim_vf = _mlp(feature_dim, vf, activation_fn)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy cuyo actor solo usa las primeras `actor_dim` entradas."""

    def __init__(self, *args, actor_dim: int = None, **kwargs):
        self.actor_dim = actor_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        actor_dim = self.actor_dim or self.features_dim
        self.mlp_extractor = AsymmetricExtractor(
            self.features_dim, actor_dim, self.net_arch, self.activation_fn)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        return data


def actor_weights(model) -> list:
    """[(W, b), ...] de las capas del actor (incluida la de salida), en float32."""
    policy = model.policy
    layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    layers.append(policy.action_net)
    return [(l.weight.detach().cpu().numpy().astype(np.float32),
             l.bias.detach().cpu().numpy().astype(np.float32)) for l in layers]


def actor_forward(weights, actor_obs: np.ndarray) -> np.ndarray:
    """Inferencia del actor con NumPy (lo mismo que haría el robot). Salida recortada a [-1, 1]."""
    x = np.asarray(actor_obs, dtype=np.float32)
    for i, (W, b) in enumerate(weights):
        x = W @ x + b
        if i < len(weights) - 1:
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0)
