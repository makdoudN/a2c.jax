""" Neural Networks.
"""

import jax
import jax.numpy as jnp
import haiku as hk

from tensorflow_probability.substrates import jax as tfp


def mlp_deterministic(
    output_size: int,
    hidden_sizes=[64, 64],
    activation="silu",
    final_tanh_activation: bool = False,
):

    if isinstance(activation, str):
        activation = getattr(jax.nn, activation)

    def forward(x):
        core = hk.nets.MLP(
            [*hidden_sizes, output_size],
            activation=activation,
            w_init=hk.initializers.Orthogonal(),
        )
        z = core(x)
        if final_tanh_activation:
            z = jnp.tanh(z)
        return z

    return forward


def mlp_multivariate_normal_diag(
    output_size: int,
    hidden_sizes=[64, 64],
    activation="silu",
    logstd_min=-10.0,
    logstd_max=1.0,
    fixed_std_value: float = 1.0,
    fixed_std: bool = False,
    state_dependent_std: bool = True,
    use_tanh_bijector: bool = False,
):
    if isinstance(activation, str):
        activation = getattr(jax.nn, activation)

    def forward(x, temperature: float = 1.0):
        core = hk.nets.MLP(
            hidden_sizes,
            activation=activation,
            activate_final=True,
            w_init=hk.initializers.Orthogonal(),
        )
        z = core(x)
        if not fixed_std:
            if state_dependent_std:
                logstd = hk.Linear(output_size)(z)
            else:
                logstd = hk.get_parameter(
                    "logstd",
                    [
                        output_size,
                    ],
                    init=lambda shape, dtype: jnp.zeros(shape, dtype),
                )
            logstd = jnp.tanh(logstd)
            logstd = logstd_min + 0.5 * (logstd_max - logstd_min) * (logstd + 1.0)
            std = jnp.exp(logstd)
        else:
            std = jnp.ones(shape=(output_size,)) * fixed_std_value
        mean = hk.Linear(output_size, w_init=hk.initializers.Orthogonal())(z)
        # TODO: Add optional bijector
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std * temperature)
    return forward


def mlp_categorical(
    output_size: int,
    hidden_sizes=[64, 64],
    activation="silu",
):
    if isinstance(activation, str):
        activation = getattr(jax.nn, activation)

    def forward(x):
        core = hk.nets.MLP(
            [*hidden_sizes, output_size],
            activation=activation,
            activate_final=False,
            w_init=hk.initializers.Orthogonal(),
        )
        z = core(x)
        return tfp.distributions.Categorical(logits=z)
    return forward


