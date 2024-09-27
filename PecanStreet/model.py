import tensorflow_probability.substrates.jax as tfp
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from functools import partial
import seaborn as sns
import tensorflow as tf
from jax import vmap
dist = tfp.distributions

from tqdm import tqdm


class seq2point(nn.Module):
    @nn.compact
    def __call__(self, X, deterministic):
        X = nn.Conv(30, kernel_size=(10,), dtype=jnp.float16)(X)
        X = nn.relu(X)
        X = nn.Conv(30, kernel_size=(8,), dtype=jnp.float16)(X)
        X = nn.relu(X)        
        X = nn.Conv(40, kernel_size=(6,), dtype=jnp.float16)(X)
        X = nn.relu(X)
        X = nn.Conv(50, kernel_size=(5,), dtype=jnp.float16)(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        X = nn.Conv(50, kernel_size=(5,), dtype=jnp.float16)(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        X = X.reshape((X.shape[0], -1))
        X = nn.Dense(1024, dtype=jnp.float16)(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        sigma = nn.softplus(nn.Dense(1, dtype=jnp.float32)(X))
        X = nn.Dense(64, dtype = jnp.float16)(X)
        X = nn.relu(X)
        mean = nn.Dense(1, dtype = jnp.float32)(X)
        return mean, sigma

    def loss_fn(self, params, X, y, deterministic=False, rng=jax.random.PRNGKey(0)):
        mean, sigma = self.apply(
            params, X, deterministic=deterministic, rngs={"dropout": rng}
        )

        def loss(mean, sigma, y):
            d = dist.Normal(loc=mean, scale=sigma)
            return -d.log_prob(y)

        return jnp.mean(jax.vmap(loss, in_axes=(0, 0, 0))(mean, sigma, y))

    



def fit(
    model,
    params,
    X,
    y,
    deterministic,
    batch_size=32,
    learning_rate=0.01,
    epochs=10,
    rng=jax.random.PRNGKey(0),
):
    opt = optax.adam(learning_rate=learning_rate)
    opt_state = opt.init(params)
    model = seq2point()
    loss_fn = partial(model.loss_fn, deterministic=deterministic)
    loss_grad_fn = jax.value_and_grad(loss_fn)
    losses = []
    total_epochs = (len(X) // batch_size) * epochs

    carry = {}
    carry["params"] = params
    carry["state"] = opt_state

    @jax.jit
    def one_epoch(carry, rng):
        params = carry["params"]
        opt_state = carry["state"]
        idx = jax.random.choice(
            rng, jnp.arange(len(X)), shape=(batch_size,), replace=False
        )
        loss_val, grads = loss_grad_fn(params, X[idx], y[idx], rng=rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        carry["params"] = params
        carry["state"] = opt_state

        return carry, loss_val

    pbar = tqdm(total=total_epochs, desc='Training Progress', unit='epoch')
    print(carry)

    carry, losses = jax.lax.scan(
        one_epoch,
        carry,
        jax.random.split(rng, total_epochs),
    )
    pbar.update(total_epochs)  # Update progress bar to 100%
    pbar.close()

    return carry["params"], losses
