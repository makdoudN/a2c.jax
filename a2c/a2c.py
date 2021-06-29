"""A2C Core functions"""

import jax
import chex
import rlax
import tree
import numpy as np
import jax.numpy as jnp
import typing

from jax import jit
from jax import vmap
from jax import grad
from jax import partial
from rlax import truncated_generalized_advantage_estimation

vmap_gae = vmap(truncated_generalized_advantage_estimation, (1, 1, None, 1), 1)


@chex.dataclass
class Data:
    """Inputs Data for computing PPO losses."""
    observation: chex.Array
    last_observation: chex.Array
    terminal: chex.Array
    reward: chex.Array
    action: chex.Array


@chex.dataclass
class Batch:
    observation: chex.Array
    action: chex.Array
    advantage: chex.Array
    returns: chex.Array


@chex.dataclass
class State:
    """ Variable State needed for the algorithm."""
    key: chex.Array
    params: chex.ArrayTree
    opt_state: chex.ArrayTree


def process_data(
    params: typing.Dict[str, typing.Any],
    data: Data,
    value_apply: typing.Callable,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    discount_factor: float = 0.99
) -> Batch:

    H, E = data['reward'].shape
    value_params = params['value']

    # -- Compute A^{GAE}: (Trunctated) Generalized Advantage Function.
    vf = value_apply(value_params, data['observation'])
    vf_last = value_apply(value_params, data['last_observation'])[None]
    values = jnp.concatenate([vf, vf_last])
    discount = data['terminal'] * discount_factor
    advantage = vmap_gae(data['reward'] * reward_scaling, discount, lambda_, values)
    chex.assert_equal_shape([vf, advantage])
    returns = advantage + vf

    # -- Flatten the data, from (H, E, *) to (B=H*E, *)
    vf = vf.reshape(H * E)
    returns = returns.reshape(H * E)
    advantage = advantage.reshape(H * E,)
    action = data['action'].reshape(H * E, -1)
    if data['action'].ndim == 2:       # is discrete [H, E]
        action = action.squeeze(-1)
    observation = data['observation']
    observation = observation.reshape(H * E, *observation.shape[2:])

    return Batch(observation=observation,
                 action=action,
                 returns=returns,
                 advantage=advantage)


@partial(jit, static_argnums=(2, 3, 4, 5))
def loss_fn(
    params: typing.Dict[str, typing.Any],
    batch: Batch,
    policy_apply: typing.Callable,
    value_apply: typing.Callable,
    value_cost: float = 0.5,
    entropy_cost: float = 1e-4,
):
    """Compute the PPO Loss Function."""
    value_params = params['value']
    policy_params = params['policy']

    # -- Loss Policy
    vf = value_apply(value_params, batch.observation)
    policy_distrib = policy_apply(policy_params, batch.observation)
    entropy = policy_distrib.entropy()
    logprob = policy_distrib.log_prob(batch.action)
    chex.assert_equal_shape([logprob, batch.advantage])
    loss_H = -entropy_cost * entropy.mean()
    loss_PI = -logprob * batch.advantage
    loss_PI = loss_PI.mean()
    loss_policy = loss_PI + loss_H

    # -- Loss value function
    chex.assert_equal_shape([vf, batch.returns])
    loss_V = rlax.l2_loss(vf, batch.returns).mean()
    loss = loss_policy + value_cost * loss_V

    metrics = dict(a2c_loss=loss, policy_loss=loss_PI, value_loss=loss_V, H_loss=loss_H)
    return loss, metrics


