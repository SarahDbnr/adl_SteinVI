import jax.scipy.stats as stats
import jax.numpy as jnp


def logp_unnormalized_posterior(x, input_1, input_2, true_output, prior_mu, prior_prec, nnet_model, treedef):
    # x are estimated parameters
    log_prior = stats.multivariate_normal.logpdf(x, prior_mu, prior_prec)
    prediction = nnet_model.apply(treedef(x), input_1, input_2).squeeze()
    log_likelihood = jnp.sum(stats.norm.logpdf(true_output, loc=prediction, scale=jnp.ones(len(prediction))))
    log_unnormalized_posterior = log_prior + log_likelihood
    return log_unnormalized_posterior
