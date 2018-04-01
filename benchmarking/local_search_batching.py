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
    init_times = []
    powers = np.arange(1, 8, 0.5).tolist()
    for power in tqdm(powers):
        points = np.random.uniform(size=(int(10 ** power), 2))

        before_init = time()
        batching_function = batching_functions.progressive_local_search(points)
        init_time = time() - before_init
        init_times += [init_time]
        batching_times += [
            timeit(
                stmt=lambda: batching_function(batch_size=10, context={'current_step': 0}, targets=points),
                number=20
            ) / 20
        ]

    trace1 = go.Scatter(x=powers, y=batching_times, name='batching time')
    trace2 = go.Scatter(x=powers, y=init_times, name='init time')
    fig = go.Figure(data=[trace1, trace2])
    plotly.offline.plot(fig)


if __name__ == '__main__':
    main()
