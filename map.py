import os

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.ndimage

UP = np.array([-1, 0])
DOWN = np.array([1, 0])
LEFT = np.array([0, -1])
RIGHT = np.array([0, 1])
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


def terrainify(array):

    deep_ocean = array < 0.3
    ocean = array < 0.4
    sea = array <= 0.5
    beach = array > 0.5
    land = array > 0.6
    inland = array > 0.8

    array[beach] = 0.7
    array[land] = 0.5
    array[inland] = 0.4
    array[sea] = 0.2
    array[ocean] = 0.15
    array[deep_ocean] = 0.1

    # land = array > 0.5
    # ocean = array <= 0.5
    # array[land] = 0.5
    # array[ocean] = 0.15

    array[0, 0] = 1
    array[1, 0] = 0

    return array


def make_map(num_islands, width, height, prob_spawn, buffer):

    _map = np.zeros([height, width])

    seed_x = np.random.randint(buffer, width - buffer, num_islands)
    seed_y = np.random.randint(buffer, height - buffer, num_islands)

    _map[seed_y, seed_x] = 1

    current_coords = list(np.array([tuple(seed_y), tuple(seed_x)]).T)
    current_coords = [tuple(array) for array in current_coords]

    used_coords = set()

    while True:
        prev_map = _map.copy()
        nearby_coords = set()
        for coord in current_coords:
            for direction in DIRECTIONS:
                new_coord = tuple(np.array(coord) + direction)
                if new_coord not in used_coords:
                    if (buffer <= new_coord[1] < width - buffer
                            and buffer <= new_coord[0] < height - buffer):
                        if np.random.rand() < prob_spawn:
                            _map[new_coord] = 1
                            nearby_coords.add(new_coord)
        used_coords.update(current_coords)
        current_coords = nearby_coords
        if np.array_equal(prev_map, _map):
            break

    sigma_x = 7
    sigma_y = 7
    sigma = [sigma_y, sigma_x]
    _map = sp.ndimage.filters.gaussian_filter(_map, sigma, mode='constant')

    _map = terrainify(_map)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(width//100, height//100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(_map, interpolation='nearest', cmap='gist_earth', aspect='auto')
    fig.savefig(os.path.join('Assets', 'map'))


make_map(2, 1200, 700, 0.9, 50)
