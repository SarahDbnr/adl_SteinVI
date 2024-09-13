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
    if misclassified_indices.size == 0:
        print("No misclassified samples.")
        misclassified_random_indices = jnp.array([])  # or handle it according to your logic
    else:
        misclassified_random_indices = jax.random.choice(subkey1, misclassified_indices, shape=(5,), replace=False)

    correctly_classified_random_indices = jax.random.choice(subkey2, correctly_classified_indices, shape=(5,), replace=False)

    # Combine the selected indices
    indices = jnp.concatenate([misclassified_random_indices, correctly_classified_random_indices])
    for idx in indices:
        if image_data:
            # Get the actual image and label
            # Determine the shape of the image
            image_shape = z_test[idx].shape

            # If the image has only two dimensions (grayscale), keep it as is. 
            # If it has three dimensions (colored), use the full shape.
            if len(image_shape) == 2:
                image = z_test[idx].reshape(image_shape)
            else:
                image = z_test[idx].reshape(image_shape)  # Automatically handles colored images (e.g., 28x28x3)

            
            true_label = y_test[idx]
            predicted_label = predicted_classes[idx]
        
        
            # Display the image and the prediction details
            # Display the image using the appropriate color map
            plt.imshow(image, cmap='gray' if image_shape[-1] == 1 or len(image_shape) == 2 else None)
            plt.title(f"True Label: {true_label}, Predicted: {predicted_label}\n")
            plt.show()
        view_probabilities_classification(averaged_precision[int(idx),:],predicted_classes[int(idx)],y_test[int(idx)])