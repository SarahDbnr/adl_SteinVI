import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

def view_missclassified(out, nnet_model, tree_def, z_test, y_test, output_size, missclassified=True):
    """
    Only for images. This function will display the image
    and show how many particles predicted each class.

    Parameters:
    - out: The output from the SVGD training, containing particles.
    - nnet_model: The neural network model used for predictions.
    - tree_def: The tree structure used by the JAX model.
    - z_test: The test dataset input.
    - y_test: The true labels for the test dataset.
    - missclassified: If you want to look at the correctly classified images or the misclassified images.
    """

    # Get predictions and precisions from all particles
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)

    # Take the mean prediction across all particles
    averaged_precision = precisions.mean(0)
    
    # Convert the averaged predictions to predicted classes
    predicted_classes = jnp.argmax(averaged_precision, axis=-1)
    
    # Find the indices where the predicted classes do or do not match the true labels
    if missclassified:
        misclassified_indices = jnp.where(predicted_classes != y_test)[0]
    else:
        misclassified_indices = jnp.where(predicted_classes == y_test)[0]
    
    for idx in misclassified_indices:
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
    
        class_votes = jnp.zeros(output_size, dtype=int)  # Assuming 10 classes
        for particle_prediction in precisions[:, idx]:
            class_votes = class_votes.at[jnp.argmax(particle_prediction)].add(1)
        
        # Display the image and the prediction details
        # Display the image using the appropriate color map
        plt.imshow(image, cmap='gray' if image_shape[-1] == 1 or len(image_shape) == 2 else None)
        plt.title(f"True Label: {true_label}, Predicted: {predicted_label}\n"
                  f"Class votes: {class_votes}")
        plt.show()

        # Print the particle vote breakdown
        print(f"Image {idx} was misclassified.")
        print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
        print("Class votes from particles:", class_votes)