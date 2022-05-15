import os

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import scipy.ndimage

UP = np.array([0, -1])
DOWN = np.array([0, 1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


def make_map(num_islands, width, height, prob_spawn):

    _map = np.zeros([height, width])

    seed_x = np.random.randint(0, width, num_islands)
    seed_y = np.random.randint(0, height, num_islands)

    _map[seed_y, seed_x] = 1

    current_coords = list(np.array([tuple(seed_x), tuple(seed_y)]).T)
    current_coords = [tuple(array) for array in current_coords]

    used_coords = set()

    while True:
        prev_map = _map.copy()
        nearby_coords = set()
        for coord in current_coords:
            for direction in DIRECTIONS:
                new_coord = tuple(np.array(coord) + direction)
                if new_coord not in used_coords:
                    if 0 <= new_coord[1] < width and 0 <= new_coord[0] < height:
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

    beach = _map <= 0.5
    land = _map <= 0.25
    sea = _map >= 0.5
    ocean = _map >= 0.75

    _map[beach] = 0.7
    _map[sea] = 0.2
    _map[land] = 0.4
    _map[ocean] = 0.15
    _map[0, 0] = 1
    _map[1, 0] = 0

    fig = plt.figure(frameon=False)
    fig.set_size_inches(width//100, height//100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(_map, interpolation='nearest', cmap='gist_earth', aspect='auto')
    fig.savefig(os.path.join('Assets', 'map'))
