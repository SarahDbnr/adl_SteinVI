import jax
import jax.numpy as jnp


# Evaluate the particles by averaging the predictions and calculating the accuracy
def evaluate_particles(out, nnet_model, tree_def, x_test, y_test):
    num_particles = len(out.particles)
    all_predictions = []

    for i in range(num_particles):
        particle_params = tree_def(out.particles[i])
        logits = nnet_model.apply(particle_params, x_test)
        probabilities = jax.nn.softmax(logits, axis=-1)
        all_predictions.append(probabilities)

    all_predictions = jnp.stack(all_predictions, axis=0)
    averaged_predictions = jnp.mean(all_predictions, axis=0)
    predicted_classes = jnp.argmax(averaged_predictions, axis=-1)
    accuracy = jnp.mean(predicted_classes == y_test)

    return averaged_predictions, accuracy


def evaluate_mse_on_test_data(particles, z_test, y_test, nnet_model, tree_def):
    # Average predictions across particles
    predictions = jax.vmap(lambda p: nnet_model.apply(tree_def(p), z_test)[:, 0])(particles).mean(0)
    mse = jnp.mean((predictions - y_test) ** 2)
    return mse


def evaluate_particles_regression(out, nnet_model, tree_def, x_test, y_test):
    predictions = jax.vmap(lambda p: nnet_model.apply(tree_def(p), x_test)[:, 0])(out.particles).mean(0)
    mse = jnp.mean((predictions - y_test) ** 2)

    print(f"Validation MSE: {mse}")
    jax.debug.print("Prediction[1:5]: {}", predictions[1:5])
    jax.debug.print("Test[1:5]: {}", y_test[1:5])
    jax.debug.print("Test[1:5]: {}", y_test.max())
    jax.debug.print("Test[1:5]: {}", y_test.min())

    return mse
