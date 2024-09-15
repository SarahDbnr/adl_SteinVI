import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from src.metrics.plots_validation_metrics import view_probabilities_classification
def view_misclassified(out, nnet_model, tree_def, z_test, y_test, key, image_data):
    """
    This function will display the image if it is image_data and will show the probabilities for the classes are distributed.
    5 random misclassified and 5 random right classified data examples.
    images will be shown.

    Parameters:
    - out: The output from the SVGD training, containing particles.
    - nnet_model: The neural network model used for predictions.
    - tree_def: The tree structure used by the JAX model.
    - z_test: The test dataset input.
    - y_test: The true labels for the test dataset.
    - key: JAX random key for initialization.
    - image_data: If the data is image data or not
    """

    # Get predictions and precisions from all particles
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)

    # Take the mean prediction across all particles
    averaged_precision = precisions.mean(0)
    
    # Convert the averaged predictions to predicted classes
    predicted_classes = jnp.argmax(averaged_precision, axis=-1)
    
    # Find the indices where the predicted classes do or do not match the true labels
    misclassified_indices = jnp.where(predicted_classes != y_test)[0]
    correctly_classified_indices = jnp.where(predicted_classes == y_test)[0]

    # Shuffle and select 5 random indices from each
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Handle misclassified indices
    if misclassified_indices.size == 0:
        print("No misclassified samples.")
        misclassified_random_indices = jnp.array([])  # or handle it according to your logic
    else:
        # Use all samples if fewer than 5, otherwise randomly select 5
        num_misclassified_samples = min(5, misclassified_indices.size)
        misclassified_random_indices = jax.random.choice(subkey1, misclassified_indices, shape=(num_misclassified_samples,), replace=False)

    # Handle correctly classified indices
    if correctly_classified_indices.size == 0:
        print("No correctly classified samples.")
        correctly_classified_random_indices = jnp.array([])  # or handle it according to your logic
    else:
        # Use all samples if fewer than 5, otherwise randomly select 5
        num_correctly_classified_samples = min(5, correctly_classified_indices.size)
        correctly_classified_random_indices = jax.random.choice(subkey2, correctly_classified_indices, shape=(num_correctly_classified_samples,), replace=False)

    # Combine the selected indices
    indices = jnp.concatenate([misclassified_random_indices, correctly_classified_random_indices])
    for idx in indices:
        if image_data:
            # Get the actual image and label
            image_shape = z_test[idx].shape
            image = z_test[idx].reshape(image_shape)
            true_label = y_test[idx]
            predicted_label = predicted_classes[idx]

            # Create a figure with two subplots: one for the image and one for the probability distribution
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Display the image in the first subplot
            ax[0].imshow(image, cmap='gray' if image_shape[-1] == 1 or len(image_shape) == 2 else None)
            ax[0].set_title(f"True Label: {true_label}, Predicted: {predicted_label}")
            ax[0].axis('off')

            # Plot the probability distribution in the second subplot
            view_probabilities_classification(precisions[:, int(idx), :], predicted_classes[int(idx)], y_test[int(idx)], ax=ax[1])

            plt.tight_layout()
            plt.show()

        else:
            # If not image data, simply show the probability distribution
            view_probabilities_classification(precisions_sample=precisions[:, int(idx), :],predicted_class= predicted_classes[int(idx)], true_class=y_test[int(idx)])