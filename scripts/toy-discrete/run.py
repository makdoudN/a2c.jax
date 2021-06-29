import gym
import jax
import tax
import yaml
import optax
import hydra
import tqdm
import jax.numpy as jnp
import numpy as np
import haiku as hk

from a2c.a2c import State
from a2c.a2c import loss_fn
from a2c.a2c import process_data
from a2c.common.utils import evaluation
from a2c.common.nn import mlp_categorical
from a2c.common.nn import mlp_deterministic
from jax import jit
from functools import partial
from omegaconf import OmegaConf
from mlrec.recorder import Recorder
from gym.vector import AsyncVectorEnv


tax.set_platform("cpu")


def save_conf(conf: OmegaConf, rec) -> None:
    yconf = OmegaConf.to_yaml(conf, resolve=True)
    print(yconf)
    rec.save(yaml.safe_load(yconf), "conf.yaml")


def setup_envs(conf):
    make_env = lambda: gym.make("CartPole-v0")
    venv = AsyncVectorEnv([make_env for _ in range(conf.num_envs)])
    env_valid = make_env()
    action_size = env_valid.action_space.n
    observation_size = env_valid.observation_space.shape[0]
    return {
        "env": venv,
        "env_valid": env_valid,
        "observation_size": observation_size,
        "action_size": action_size,
    }


def setup(conf):
    rng = jax.random.PRNGKey(conf.seed)
    envs_info = setup_envs(conf)
    action_size = envs_info["action_size"]
    observation_size = envs_info["observation_size"]
    dummy_observation = jnp.zeros((observation_size))

    value_def = lambda x: mlp_deterministic(1, **conf.value_kwargs)(x).squeeze(-1)
    value_def = hk.transform(value_def)
    value_def = hk.without_apply_rng(value_def)
    value_opt = getattr(optax, conf.value_opt)(**conf.value_opt_kwargs)

    policy_def = lambda x: mlp_categorical(action_size, **conf.policy_kwargs)(x)
    policy_def = hk.transform(policy_def)
    policy_def = hk.without_apply_rng(policy_def)
    policy_opt = getattr(optax, conf.policy_opt)(**conf.policy_opt_kwargs)

    rng, rng_policy, rng_value = jax.random.split(rng, 3)
    value_params = value_def.init(rng_value, dummy_observation)
    policy_params = policy_def.init(rng_policy, dummy_observation)
    value_opt_state = value_opt.init(value_params)
    policy_opt_state = policy_opt.init(policy_params)

    params = {"policy": policy_params, "value": value_params}
    opt_state = {"policy": policy_opt_state, "value": value_opt_state}

    process_data_to_batch = partial(process_data, value_apply=value_def.apply)
    loss = partial(loss_fn, value_apply=value_def.apply, policy_apply=policy_def.apply)

    @jit
    def update_fn(state, inputs):
        """ Generic Update function """
        g, metrics = jax.grad(loss, has_aux=True)(state.params, inputs)

        updates, value_opt_state = value_opt.update(g["value"], state.opt_state["value"])
        value_params = jax.tree_multimap(
            lambda p, u: p + u, state.params["value"], updates
        )

        updates, policy_opt_state = policy_opt.update(g["policy"], state.opt_state["policy"])
        policy_params = jax.tree_multimap(
            lambda p, u: p + u, state.params["policy"], updates
        )

        params = state.params
        params = dict(policy=policy_params, value=value_params)
        opt_state = state.opt_state
        opt_state = dict(policy=policy_opt_state, value=value_opt_state)

        state = state.replace(params=params, opt_state=opt_state)
        return state, metrics

    def interaction(env, horizon: int = 10, seed: int = 42):
        steps = 0
        rng = jax.random.PRNGKey(seed)
        observation, buf = env.reset(), []
        num_envs = observation.shape[0]
        params = yield

        while True:
            for _ in range(horizon):
                steps += num_envs
                rng, rng_action = jax.random.split(rng)
                action = jit(policy_def.apply)(params, observation).sample(
                    seed=rng_action
                )
                observation_next, reward, done, info = env.step(action)
                buf.append(
                    {
                        "observation": observation,
                        "reward": reward,
                        "terminal": 1.0 - done,
                        "action": action,
                    }
                )
                observation = observation_next.copy()

            data = jit(tax.reduce)(buf)
            data["last_observation"] = observation
            info = {'steps': steps}
            params = yield data, info
            buf = []

    interaction_step = interaction(envs_info["env"], horizon=conf.horizon, seed=conf.seed + 1)
    interaction_step.send(None)

    def make_policy(params):
        fn = lambda rng, x: policy_def.apply(params, x).sample(seed=rng)
        return jit(fn)

    return State(key=rng, params=params, opt_state=opt_state), {
        "interaction": interaction_step,
        "evaluation": lambda rng, params: evaluation(
            rng, envs_info["env_valid"], make_policy(params)
        ),
        "update": jit(
            lambda state, data: update_fn(state, process_data_to_batch(state.params, data))
        ),
    }


@hydra.main(config_path=".", config_name="conf")
def main(conf):
    rng = jax.random.PRNGKey(conf.seed)
    rec = Recorder(output_dir=".")
    save_conf(conf, rec)

    state, a2c_info = setup(conf)
    ncycles = conf.maxsteps // conf.horizon // conf.num_envs
    ncyles_per_epoch = ncycles // conf.nepochs

    for e in range(conf.nepochs):
        S = tax.Store()
        for _ in tqdm.trange(ncyles_per_epoch):
            data, info_interaction = a2c_info['interaction'].send(state.params['policy'])
            state, info_update = a2c_info['update'](state, data)
            S.add(**info_update, **info_interaction)
        rng, rng_eval = jax.random.split(rng)
        info_eval = a2c_info['evaluation'](rng_eval, state.params['policy'])
        info = S.get()
        info.update(info_eval)
        print(info)
if __name__ == "__main__":
    main()
