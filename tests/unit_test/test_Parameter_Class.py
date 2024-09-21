from optax import adam
from stein_vi.Classes.Parameter_Class import Parameter


def test_parameter_init():
    """Test the initialization of the Parameter class."""
    # given
    optimizer = adam(learning_rate=0.001)

    # when
    param = Parameter(
        optimizer=optimizer,
        early_stopping=True,
        image_data=True,
        batch_size=64,
        particle_batch_size=10,
        num_particles=20,
        num_iterations=100,
        learning_rate=0.001
    )

    # then
    assert param.optimizer == optimizer, "Optimizer should be set correctly"
    assert param.batch_size == 64, "Batch size should be set correctly"
    assert param.particle_batch_size == 10, "Particle batch size should be set correctly"
    assert param.num_particles == 20, "Number of particles should be set correctly"
    assert param.num_iterations == 100, "Number of iterations should be set correctly"
    assert param.stopped_at_iteration == 100, "Stopped iteration should be initialized to num_iterations"
    assert param.early_stopping is True, "Early stopping flag should be set correctly"
    assert param.learning_rate == 0.001, "Learning rate should be set correctly"
    assert param.image_data is True, "Image data flag should be set correctly"

    assert param.kernel_length == 0.05, "Default kernel length should be 0.05"
    assert param.warm_up_iterations_early_stopping == 30, "Default warm up iterations should be 30"
    assert param.patience_early_stopping == 15, "Default patience should be 15"
    assert param.min_delta_early_stopping == 0.005, "Default min delta for early stopping should be 0.005"


def test_parameter_set_early_stopping():
    """Test the set_early_stopping function."""

    # given
    optimizer = adam(learning_rate=0.001)

    param = Parameter(
        optimizer=optimizer,
        early_stopping=True,
        image_data=False,
        batch_size=32,
        particle_batch_size=5,
        num_particles=10,
        num_iterations=50,
        learning_rate=0.01
    )

    # when
    param.set_early_stopping(
        warm_up_iterations=50,
        patience=10,
        min_delta=0.01
    )

    # then
    assert param.warm_up_iterations_early_stopping == 50, "Warm up iterations should be updated to 50"
    assert param.patience_early_stopping == 10, "Patience should be updated to 10"
    assert param.min_delta_early_stopping == 0.01, "Min delta should be updated to 0.01"
