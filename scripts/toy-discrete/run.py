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
from common import setup


tax.set_platform("cpu")


def save_conf(conf: OmegaConf, rec) -> None:
    yconf = OmegaConf.to_yaml(conf, resolve=True)
    print(yconf)
    rec.save(yaml.safe_load(yconf), "conf.yaml")


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
        rec.write(info)


if __name__ == "__main__":
    main()
