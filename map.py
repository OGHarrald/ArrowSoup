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

    deep_ocean = array < 0.05
    ocean = array < 0.2
    sea = array <= 0.4
    beach = array > 0.4
    land = array > 0.5
    inland = array > 0.8
    mountain = array > 0.91
    tip = array > 0.95

    array[beach] = 0.7
    array[land] = 0.55
    array[inland] = 0.5
    array[mountain] = 0.45
    array[tip] = 0.4
    array[sea] = 0.2
    array[ocean] = 0.15
    array[deep_ocean] = 0.1

    array[0, 0] = 1
    array[1, 0] = 0

    return array


def save_map(_map, width, height, cmap, filename):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(width//100, height//100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(_map, interpolation='nearest', cmap=cmap, aspect='auto')
    fig.savefig(os.path.join('Assets', filename))


def make_map(num_islands, width, height, prob_spawn, buffer):

    _map = np.zeros([height, width])

    seed_x = np.random.randint(buffer, width - buffer, num_islands)
    seed_y = np.random.randint(buffer, height - buffer, num_islands)

    _map[seed_y, seed_x] = 1

    current_coords = list(np.array([tuple(seed_y), tuple(seed_x)]).T)
    current_coords = [tuple(array) for array in current_coords]

    used_coords = set()

    i = 1

    while True:
        prev_map = _map.copy()
        nearby_coords = set()
        for coord in current_coords:
            for direction in DIRECTIONS:
                new_coord = tuple(np.array(coord) + direction)
                if new_coord not in used_coords:
                    if (buffer <= new_coord[1] < width - buffer
                            and buffer <= new_coord[0] < height - buffer):
                        new_coord_neighbours = [tuple(np.array(new_coord) + direction)
                                                for direction in DIRECTIONS]
                        new_coord_neighbours = [c for c in new_coord_neighbours
                                                if buffer <= c[0] < height - buffer
                                                and buffer <= c[1] < width - buffer]

                        land_total = int(sum([_map[c] for c in new_coord_neighbours]))

                        for _ in range(land_total + 1):
                            if np.random.rand() < prob_spawn:
                                _map[new_coord] = 1
                                nearby_coords.add(new_coord)
                                break

        used_coords.update(current_coords)
        current_coords = nearby_coords
        if np.array_equal(prev_map, _map):
            break
        if i == 800:
            break
        i += 1

    sigma_x = 7
    sigma_y = 7
    sigma = [sigma_y, sigma_x]
    _map = sp.ndimage.filters.gaussian_filter(_map, sigma, mode='constant')

    _map = terrainify(_map)

    save_map(_map, width, height, 'gist_earth', 'map')
    save_map(_map, width, height, 'bone', 'minimap')

make_map(20, 1200, 700, 0.2455, 30)
