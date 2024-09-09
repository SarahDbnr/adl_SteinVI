import jax.scipy.stats as stats
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm


def get_posteriori(nnet_model, tree_def, regression):
    """
    Selects and compiles the log-posterior function for a neural network model based on the type of problem (regression or multinomial classification).

    Args:
        nnet_model (flax.linen.Module): The neural network model to use for predictions.
        tree_def (jax.tree_util.PyTreeDef): The tree definition for parameter transformation.
        regression (bool): Flag to determine whether the model is used for regression (True) or classification (False).

    Returns:
        Callable: A JIT-compiled function that computes the log-posterior for given parameters and data.
    """
    if regression:
        @jax.jit
        def logp_model(params, dz, dy):
            return logp_unnormalized_posterior_regression(
                params,
                nnet_model=nnet_model,
                dz=dz,
                dy=dy,
                treedef=tree_def,
            )
    else:
        @jax.jit
        def logp_model(params, dz, dy):
            return logp_unnormalized_posterior_mulitnomial(
                params,
                nnet_model=nnet_model,
                dz=dz,
                dy=dy,
                treedef=tree_def,
            )
    return logp_model


def logp_unnormalized_posterior_regression(params, dz, dy, nnet_model, treedef):
    """
    Computes the log of the unnormalized posterior probability for a regression problem, using a Gaussian prior and likelihood.

    Args:
        params (jax.numpy.ndarray): Neural network parameters.
        dz (jax.numpy.ndarray): Input features.
        dy (jax.numpy.ndarray): Observed target values.
        nnet_model (flax.linen.Module): The neural network model to use for predictions.
        treedef (jax.tree_util.PyTreeDef): The tree definition for parameter transformation.

    Returns:
        float: The log of the unnormalized posterior probability.
    """
    log_prior = jnp.sum(norm.logpdf(params, loc=0, scale=1))

    prediction_mean, prediction_var_score = jnp.split(nnet_model.apply(treedef(params), dz), 2, axis=-1)
    location = prediction_mean.squeeze()
    scale = jax.vmap(lambda p: link_function(p))(prediction_var_score.squeeze())

    log_likelihood = jnp.sum(norm.logpdf(dy, loc=location, scale=scale))

    return log_prior + log_likelihood


def link_function(x):
    """
    Transforms an input to ensure the scale parameter is positive. Uses the function log(1 + abs(x)).

    Args:
        x (jax.numpy.ndarray): Input value(s) that represent pre-scale parameters.

    Returns:
        jax.numpy.ndarray: Transformed value(s) to ensure positivity.
    """
    return jnp.log(1 + jnp.abs(x))


def logp_unnormalized_posterior_mulitnomial(params, dz, dy, nnet_model, treedef):
    """
    Computes the log of the unnormalized posterior probability for a multinomial classification problem, using a Gaussian prior and a multinomial likelihood.

    Args:
        params (jax.numpy.ndarray): Neural network parameters.
        dz (jax.numpy.ndarray): Input features.
        dy (jax.numpy.ndarray): Observed target values (class labels).
        nnet_model (flax.linen.Module): The neural network model to use for predictions.
        treedef (jax.tree_util.PyTreeDef): The tree definition for parameter transformation.

    Returns:
        float: The log of the unnormalized posterior probability.
    """
    log_prior = jnp.sum(stats.norm.logpdf(params, loc=0, scale=1))
    dy = dy.ravel()
    logits = nnet_model.apply(treedef(params), dz)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_likelihood = jnp.sum(log_probs[jnp.arange(dy.shape[0]), dy])
    log_unnormalized_posterior = log_prior + log_likelihood
    return log_unnormalized_posterior
