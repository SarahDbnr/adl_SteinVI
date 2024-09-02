import jax.scipy.stats as stats
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm


def get_posteriori(nnet_model, tree_def, regression):
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
    # Calculate the log-prior (Gaussian prior on the weights)
    log_prior = jnp.sum(norm.logpdf(params, loc=0, scale=1))

    # Get predictions from the neural network
    prediction_mean, prediction_var_score = jnp.split(nnet_model.apply(treedef(params), dz), 2, axis=-1)
    location = prediction_mean.squeeze()
    # set maximal standard deviation
    max_scale = jnp.abs(location.mean()) * 2
    scale = jax.vmap(lambda p: link_function(p, max_scale))(prediction_var_score.squeeze())
    log_likelihood = jnp.sum(norm.logpdf(dy, loc=location, scale=scale))

    #jax.debug.print("\nPrecision: {prediction_var_score}Scale: {scale}, Prior: {log_prior}, Likelihood: {"
    #                "log_likelihood}", prediction_var_score=prediction_var_score.squeeze(), scale=scale,
    #                log_prior=log_prior, log_likelihood=log_likelihood)

    return log_prior + log_likelihood


def link_function(x, max_scale):
    return jnp.clip(jnp.log(1 + jnp.exp(0.00001 * x)), a_max=max_scale, a_min=0.1)


def logp_unnormalized_posterior_mulitnomial(params, dz, dy, nnet_model, treedef):
    # Calculate the log-prior (Gaussian prior on the weights)
    log_prior = jnp.sum(stats.norm.logpdf(params, loc=0, scale=1))

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
