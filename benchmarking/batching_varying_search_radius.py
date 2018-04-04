"""
Benchmark the speed of the batching function (for developing something with millisecond latency)
"""
from timeit import timeit
from time import time

import numpy as np
import plotly
import plotly.graph_objs as go
from tqdm import tqdm

import batching_functions


def main():
    batching_times = []
    distances = np.arange(0, 1, 0.01)
    points = np.random.uniform(size=(int(10 ** 7), 2))
    batching_function = batching_functions.progressive_local_search(points)
    for distance in tqdm(distances):
        batching_times += [
            timeit(
                stmt=lambda: batching_function(batch_size=10, context={
                    # 'current_step': 1e10,
                    'current_step': 0,
                    'average_l2_loss': distance
                }, targets=points),
                number=20
            ) / 20
        ]

    trace1 = go.Scatter(x=distances, y=batching_times, name='batching time')
    fig = go.Figure(data=[trace1])
    plotly.offline.plot(fig)


if __name__ == '__main__':
    main()
