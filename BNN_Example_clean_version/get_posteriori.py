import jax.scipy.stats as stats
import jax.numpy as jnp
import jax

@jax.jit
def logp_unnormalized_posterior_regession(x, input_1, input_2, true_output, prior_mu, prior_prec, nnet_model, treedef):
    # Vectorize the log_prior calculation
    log_prior = jnp.sum(stats.norm.logpdf(x, prior_mu, 1))
    
    # Use vmap to vectorize the prediction
    prediction = jax.vmap(lambda i1, i2: nnet_model.apply(treedef(x), i1, i2).squeeze())(input_1, input_2)
    
    # Vectorize log_likelihood calculation
    log_likelihood = jnp.sum(stats.norm.logpdf(true_output, loc=prediction, scale=1.0))
    
    return log_prior + log_likelihood

jax.jit
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

# Helper function to create a jitted version of the log posterior
def make_jitted_logp(nnet_model, tree_def, z_train, y_train, prior_mu):
    @jax.jit
    def jitted_logp(params):
        return logp_unnormalized_posterior_mulitnomial(
            params,
            dz=z_train,
            dy=y_train,
            prior_mu=prior_mu,
            nnet_model=nnet_model,
            treedef=tree_def
        )
    return jitted_logp
