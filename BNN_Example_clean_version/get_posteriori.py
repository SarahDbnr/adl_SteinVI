import jax.scipy.stats as stats
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm


def get_posteriori(nnet_model, tree_def, prior_mu, regression):
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
                prior_mu=prior_mu,
                treedef=tree_def,
            )
    return logp_model


def logp_unnormalized_posterior_regression(params, dz, dy, nnet_model, treedef):
    # Calculate log prior using Gamma distribution
    log_prior = jnp.sum(norm.logpdf(params, loc=0, scale=1))

    # Get predictions from the neural network
    prediction_mean, prediction_var_score = jnp.split(nnet_model.apply(treedef(params), dz), 2, axis=-1)
    location = prediction_mean.squeeze()
    scale = jnp.exp(0.000001 * prediction_var_score.squeeze())  # jax.nn.sigmoid(prediction_var_score.squeeze()) + 1

    log_likelihood = jnp.sum(norm.logpdf(dy, loc=location, scale=scale))

    # Regularize the NNET
    l2_loss = 0.1 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

    return log_prior + log_likelihood + l2_loss


# jax.jit # TODO why cant we use jax.jit
def logp_unnormalized_posterior_mulitnomial(params, dz, dy, prior_mu, nnet_model, treedef):
    # Calculate the log-prior (Gaussian prior on the weights)
    log_prior = jnp.sum(stats.norm.logpdf(params, prior_mu, 1))

    # Ensure dy is flattened
    dy = dy.ravel()

    # Apply the neural network model
    logits = nnet_model.apply(treedef(params), dz)

    # Calculate log-likelihood using log_softmax for numerical stability
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_likelihood = jnp.sum(log_probs[jnp.arange(dy.shape[0]), dy])

    # Calculate the log unnormalized posterior
    log_unnormalized_posterior = log_prior + log_likelihood
    return log_unnormalized_posterior
