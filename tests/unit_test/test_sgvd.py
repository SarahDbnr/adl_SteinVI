from fixtures import stein_vi_regression_example

from stein_vi.algorithm.svgd import set_up_svgd

def test_set_up_svgd(stein_vi_regression_example):
    # when
    set_up_svgd(stein_vi_regression_example)
    # then
    stein_vi_regression_example.state