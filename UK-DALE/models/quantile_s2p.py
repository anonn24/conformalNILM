from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
dist = tfp.distributions

class QuantileRegression(nn.Module):
    alpha: float

    @nn.compact
    def __call__(self, X, deterministic):
        X = nn.Conv(30, kernel_size=(10,))(X)
        X = nn.relu(X)
        X = nn.Conv(30, kernel_size=(8,))(X)
        X = nn.relu(X)        
        X = nn.Conv(40, kernel_size=(6,))(X)
        X = nn.relu(X)
        X = nn.Conv(50, kernel_size=(5,))(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        X = nn.Conv(50, kernel_size=(5,))(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        X = X.reshape((X.shape[0], -1))
        X = nn.Dense(1024)(X)
        X = nn.relu(X)
        X = nn.Dropout(rate=0.2, deterministic=deterministic)(X)
        quantile = nn.Dense(1)(X)
        return quantile

    def loss_fn(self, params, X, y, deterministic=False, rng=jax.random.PRNGKey(0)):
        quantile = self.apply(
            params, X, deterministic=deterministic, rngs={"dropout": rng}
        )

        def loss(quantile, y, alpha):
            quantile_loss = jnp.maximum(alpha * (y - quantile), (1 - alpha) *(quantile - y))
            return jnp.mean(quantile_loss)

        return jnp.mean(jax.vmap(loss, in_axes=(0, 0, None))(quantile, y, self.alpha))
