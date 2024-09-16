from stein_vi.algorithm.model_training import train_general_algorithm
from stein_vi.algorithm.svgd import set_up_svgd


def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"):

    if algorithm == "svgd":
        steinvi = set_up_svgd(steinvi)

    steinvi = train_general_algorithm(
        steinvi=steinvi,
        dataset=dataset,
        key=key
    )

    return steinvi
