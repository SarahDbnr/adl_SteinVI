import pytest
from stein_vi.Classes.Handler_Class import Handler

def test_handler_init():
    """Test the initialization of the Handler class."""
    # when then
    handler = Handler(rf_comparison=True)
    assert handler.rf_comparison, "rf_comparison should be set to True during initialization"
    
    # when then 
    handler = Handler()
    assert handler.rf_comparison is False, "rf_comparison should be False by default"


def test_set_training_print_mode():
    """Test the set_training_print_mode function."""
    #given
    handler = Handler()
    # when
    handler.set_training_print_mode('full')
    #then 
    assert handler._full_training_print, "Full training print should be enabled"
    assert handler._reduced_training_print is False, "Reduced training print should be disabled"
    assert handler._no_training_print is False, "No training print should be disabled"
    # when 
    handler.set_training_print_mode('reduced')
    # then
    assert handler._full_training_print is False, "Full training print should be disabled"
    assert handler._reduced_training_print, "Reduced training print should be enabled"
    assert handler._no_training_print is False, "No training print should be disabled"
    # when
    handler.set_training_print_mode('none')
    # then
    assert handler._full_training_print is False, "Full training print should be disabled"
    assert handler._reduced_training_print is False, "Reduced training print should be disabled"
    assert handler._no_training_print, "No training print should be enabled"
    with pytest.raises(ValueError, match="Invalid mode. Choose from 'full', 'reduced', or 'none'."):
        handler.set_training_print_mode('invalid_mode')


def test_set_evaluation_mode():
    """Test the set_evaluation_mode function."""
    # given 
    handler = Handler()
    # when
    handler.set_evaluation_mode('full')
    # then
    assert handler._full_evaluation, "Full evaluation should be enabled"
    assert handler._minimal_evaluation is False, "Minimal evaluation should be disabled"
    # when
    handler.set_evaluation_mode('minimal')
    # then
    assert handler._full_evaluation is False, "Full evaluation should be disabled"
    assert handler._minimal_evaluation, "Minimal evaluation should be enabled"

    with pytest.raises(ValueError, match="Invalid mode. Choose from 'full', or 'minimal'."):
        handler.set_evaluation_mode('invalid_mode')

