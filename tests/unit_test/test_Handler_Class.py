import pytest
from stein_vi.Classes.Handler_Class import Handler

def test_handler_init():
    """Test the initialization of the Handler class."""
    
    # Test with rf_comparison set to True
    handler = Handler(rf_comparison=True)
    assert handler.rf_comparison is True, "rf_comparison should be set to True during initialization"
    
    # Test with rf_comparison set to False (default)
    handler = Handler()
    assert handler.rf_comparison is False, "rf_comparison should be False by default"


def test_set_training_print_mode():
    """Test the set_training_print_mode function."""

    handler = Handler()

    handler.set_training_print_mode('full')
    assert handler._full_training_print, "Full training print should be enabled"
    assert handler._reduced_training_print is False, "Reduced training print should be disabled"
    assert handler._no_training_print is False, "No training print should be disabled"

    handler.set_training_print_mode('reduced')
    assert handler._full_training_print is False, "Full training print should be disabled"
    assert handler._reduced_training_print, "Reduced training print should be enabled"
    assert handler._no_training_print is False, "No training print should be disabled"

    handler.set_training_print_mode('none')
    assert handler._full_training_print is False, "Full training print should be disabled"
    assert handler._reduced_training_print is False, "Reduced training print should be disabled"
    assert handler._no_training_print, "No training print should be enabled"

    with pytest.raises(ValueError, match="Invalid mode. Choose from 'full', 'reduced', or 'none'."):
        handler.set_training_print_mode('invalid_mode')


def test_set_evaluation_mode():
    """Test the set_evaluation_mode function."""

    handler = Handler()

    handler.set_evaluation_mode('full')
    assert handler._full_evaluation is True, "Full evaluation should be enabled"
    assert handler._minimal_evaluation is False, "Minimal evaluation should be disabled"

    handler.set_evaluation_mode('minimal')
    assert handler._full_evaluation is False, "Full evaluation should be disabled"
    assert handler._minimal_evaluation is True, "Minimal evaluation should be enabled"

    with pytest.raises(ValueError, match="Invalid mode. Choose from 'full', or 'minimal'."):
        handler.set_evaluation_mode('invalid_mode')

