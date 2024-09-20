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

    # Test setting 'full' print mode
    handler.set_training_print_mode('full')
    assert handler._full_training_print is True, "Full training print should be enabled"
    assert handler._reduced_training_print is False, "Reduced training print should be disabled"
    assert handler._no_training_print is False, "No training print should be disabled"
    
    # Test setting 'reduced' print mode
    handler.set_training_print_mode('reduced')
    assert handler._full_training_print is False, "Full training print should be disabled"
    assert handler._reduced_training_print is True, "Reduced training print should be enabled"
    assert handler._no_training_print is False, "No training print should be disabled"
    
    # Test setting 'none' print mode
    handler.set_training_print_mode('none')
    assert handler._full_training_print is False, "Full training print should be disabled"
    assert handler._reduced_training_print is False, "Reduced training print should be disabled"
    assert handler._no_training_print is True, "No training print should be enabled"
    
    # Test invalid mode
    with pytest.raises(ValueError, match="Invalid mode. Choose from 'full', 'reduced', or 'none'."):
        handler.set_training_print_mode('invalid_mode')


def test_set_evaluation_mode():
    """Test the set_evaluation_mode function."""
    
    handler = Handler()

    # Test setting 'full' evaluation mode
    handler.set_evaluation_mode('full')
    assert handler._full_evaluation is True, "Full evaluation should be enabled"
    assert handler._minimal_evaluation is False, "Minimal evaluation should be disabled"
    
    # Test setting 'minimal' evaluation mode
    handler.set_evaluation_mode('minimal')
    assert handler._full_evaluation is False, "Full evaluation should be disabled"
    assert handler._minimal_evaluation is True, "Minimal evaluation should be enabled"
    
    # Test invalid mode
    with pytest.raises(ValueError, match="Invalid mode. Choose from 'full', or 'minimal'."):
        handler.set_evaluation_mode('invalid_mode')


def test_minimal_evaluation_property():
    """Test the minimal_evaluation property."""
    
    handler = Handler()

    # Test when minimal evaluation is not active
    handler.set_evaluation_mode('full')
    assert handler.minimal_evaluation is False, "minimal_evaluation should return False when full evaluation is active"
    
    # Test when minimal evaluation is active
    handler.set_evaluation_mode('minimal')
    assert handler.minimal_evaluation is True, "minimal_evaluation should return True when minimal evaluation is active"

