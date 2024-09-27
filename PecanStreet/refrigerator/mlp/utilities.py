import tensorflow_probability.substrates.jax as tfp
import jax
import jax.numpy as jnp
import seaborn as sns

dist = tfp.distributions

import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as st


def rmse(y,yhat):
  def rmse_loss(y,yhat):
      return (y-yhat)**2
  return jnp.sqrt(jnp.mean(jax.vmap(rmse_loss,in_axes=(0,0))(y,yhat)))

def NLL(mean,sigma,y):
    def loss_fn(mean, sigma, y):
      d = dist.Normal(loc=mean, scale=sigma)
      return -d.log_prob(y)
    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, 0))(mean, sigma, y))
    
def mae(y,yhat):
  def mae_loss(y,yhat):
      return jnp.abs(y-yhat)
  return jnp.mean(jax.vmap(mae_loss,in_axes=(0,0))(y,yhat))

def ace(dataframe):
    """
    dataframe : pandas dataframe with Ideal and Counts as column for regression calibration
    It can be directly used as 2nd output from calibration_regression in plot.py 
    """
    def rmse_loss(y,yhat):
      return jnp.abs(y-yhat)
    return jnp.mean(jax.vmap(rmse_loss,in_axes=(0,0))(dataframe['Ideal'].values,dataframe['Counts'].values))

def gmm_mean_var(means_stack, sigmas_stack):
    means = jnp.stack(means_stack)
    final_mean = means.mean(axis=0)
    sigmas = jnp.stack(sigmas_stack)
    final_sigma = jnp.sqrt((sigmas**2 + means ** 2).mean(axis=0) - final_mean ** 2)
    return final_mean, final_sigma