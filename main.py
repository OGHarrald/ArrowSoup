import os
import math
import random

import pygame
import numpy as np
from datetime import datetime, timedelta

from simulator import Player, make_tracker, metrics
from map import make_map

pygame.font.init()  # initialise fonts
pygame.mixer.init()  # initialise sound

WIDTH, HEIGHT = 1200, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 60

pygame.display.set_caption("Arrow Soup")

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0, 15)
BLUE = (0, 0, 255)
ORANGE = (255, 125, 0)
BLACK = (0, 0, 0)

DETECTION_SIZE = 5
PLAYER_SIZE = 15
LINE_DATA_SIZE = 100
DETECTION_DATA_SIZE = 10
DESTINATION_WIDTH = 50
DESTINATION_HEIGHT = 10

VEL = 1
ROT_SPEED = np.radians(5)
INITIAL_ORIENT = np.radians(0)

DETECTION_PERIOD = 30
NUM_SENSORS = 10
MIN_RANGE = 20
MAX_RANGE = 300

DESTINATION_REACHED = pygame.USEREVENT + 1

WIN_FONT = pygame.font.SysFont('calibri', 100)
SCORE_FONT = pygame.font.SysFont('calibri', 50)
DESTINATIONS_FONT = pygame.font.SysFont('calibri', 5)

NUM_ISLANDS = 20
PROB_SPAWN = 0.5

NUM_DESTINATIONS = 3


def player_move(player, keys_pressed):

    if keys_pressed[pygame.K_LEFT]:
        player.orientation += player.rot_speed
    if keys_pressed[pygame.K_RIGHT]:
        player.orientation -= player.rot_speed
    if keys_pressed[pygame.K_UP]:
        new_vel = player.vel + 0.1
        if new_vel < 10:
            player.vel += 0.1
    if keys_pressed[pygame.K_DOWN]:
        new_vel = player.vel - 0.1
        if new_vel >= 0.1:
            player.vel -= 0.1

    player.move()


def handle_destination(player, destination):

    if player.colliderect(destination):
        pygame.event.post(pygame.event.Event(DESTINATION_REACHED))


def draw_player(player, line_data_size):

    plot_history = player.loc_history[-line_data_size:]
    for coord, next_coord in zip(plot_history[:-1],
                                 plot_history[1:]):
        pygame.draw.line(WIN, WHITE, coord, next_coord)

    arrow_left_x = player.x + PLAYER_SIZE * np.cos(
        -player.orientation + np.radians(90))
    arrow_left_y = player.y + PLAYER_SIZE * np.sin(
        -player.orientation + np.radians(90))
    arrow_right_x = player.x + PLAYER_SIZE * np.cos(
        -player.orientation - np.radians(90))
    arrow_right_y = player.y + PLAYER_SIZE * np.sin(
        -player.orientation - np.radians(90))
    front_x = player.x + PLAYER_SIZE / 2 * np.cos(player.orientation)
    front_y = player.y - PLAYER_SIZE / 2 * np.sin(player.orientation)

    pygame.draw.polygon(WIN,
                        WHITE,
                        [(front_x, front_y),
                         (arrow_left_x, arrow_left_y),
                         (arrow_right_x, arrow_right_y)])


def draw_sensor(sensor):

    x, y = sensor.position
    _range = sensor.max_range
    sensor_rect = pygame.Rect(x - _range, y - _range, 2 * _range, 2 * _range)
    pygame.draw.ellipse(WIN, RED, sensor_rect, 1)
    sensor_rect = pygame.Rect(x - 1, y - 1, 2, 2)
    pygame.draw.ellipse(WIN, RED, sensor_rect)


def draw_track(track, line_data_size):

    plot_track = track[-line_data_size:]
    for state1, state2 in zip(plot_track[:-1], plot_track[1:]):
        start_pos = state1.state_vector[(0, 2), :]
        end_pos = state2.state_vector[(0, 2), :]
        pygame.draw.line(WIN, ORANGE, start_pos, end_pos)


def draw_detection(detection, detection_data_size):

    centre = detection.measurement_model.inverse_function(detection)
    centre = centre[(0, 2), :].flatten()
    top_left = centre + DETECTION_SIZE * np.array([-1, -1])
    top_right = centre + DETECTION_SIZE * np.array([1, -1])
    bottom_left = centre + DETECTION_SIZE * np.array([-1, 1])
    bottom_right = centre + DETECTION_SIZE * np.array([1, 1])
    pygame.draw.line(WIN, GREEN, top_left, bottom_right)
    pygame.draw.line(WIN, GREEN, bottom_left, top_right)


def draw_window(_map, player, destinations, destinations_reached,
                sensors, tracks, all_detections,
                display_all: bool):

    if display_all:
        line_data_size = detection_data_size = 0
    else:
        line_data_size = LINE_DATA_SIZE
        detection_data_size = DETECTION_DATA_SIZE

    WIN.blit(_map, (0, 0))

    destinations_text = SCORE_FONT.render(
        f"Destinations: {destinations_reached}/{NUM_DESTINATIONS}", 1, WHITE)
    WIN.blit(destinations_text, (0, 0))

    draw_player(player, line_data_size)

    for destination in destinations:
        pygame.draw.rect(WIN, WHITE, destination)

    for sensor in sensors:
        draw_sensor(sensor)

    for track in tracks:
        draw_track(track, line_data_size)

    for detections in all_detections[-detection_data_size:]:
        for detection in detections:
            draw_detection(detection, detection_data_size)

    pygame.display.update()


def random_destination():

    destination_x = (WIDTH - DESTINATION_WIDTH) * random.random()
    destination_y = (HEIGHT - DESTINATION_HEIGHT) * random.random()
    destination = pygame.Rect(destination_x, destination_y,
                              DESTINATION_WIDTH, DESTINATION_HEIGHT)

    return destination


def random_sensors():

    sensor_x = np.random.randint(0, WIDTH, NUM_SENSORS)
    sensor_y = np.random.randint(0, HEIGHT, NUM_SENSORS)
    senor_r = np.random.randint(MIN_RANGE, MAX_RANGE, NUM_SENSORS)
    sensors = np.array([sensor_x, sensor_y, senor_r]).T

    return sensors


def main():

    make_map(NUM_ISLANDS, WIDTH, HEIGHT, PROB_SPAWN)
    _map = pygame.transform.scale(
        pygame.image.load(os.path.join('Assets', 'map.png')), (WIDTH, HEIGHT))

    player = Player(WIDTH, HEIGHT, VEL, INITIAL_ORIENT, ROT_SPEED,
                    WIDTH//10, (9*HEIGHT)//10, 5, 5)

    destination = random_destination()
    destination_history = [destination]

    sensors = random_sensors()
    
    tracker = make_tracker(sensors, DETECTION_PERIOD)

    all_detections = list()

    destinations_reached = 0

    clock = pygame.time.Clock()
    run = True
    time = datetime.now()
    turn = 0
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            if event.type == DESTINATION_REACHED:
                destinations_reached += 1
                if destinations_reached == NUM_DESTINATIONS:
                    end_game(_map, player, destination_history, destinations_reached,
                             tracker, tracker.tracks, all_detections)
                    run = False
                destination = random_destination()
                destination_history.append(destination)
                break
        if not run:
            break

        keys_pressed = pygame.key.get_pressed()

        player_move(player, keys_pressed)

        if turn % DETECTION_PERIOD == 0:
            _, tracks = tracker.track(time, player, True)
        else:
            _, tracks = tracker.track(time, player, False)

        all_detections = tracker.all_detections

        draw_window(_map, player, [destination], destinations_reached,
                    tracker.sensors, tracks, all_detections,
                    display_all=False)
        handle_destination(player, destination)
        time += timedelta(seconds=1)

        turn += 1

    main()  # start game again


def end_game(_map, player, destinations, destinations_reached,
             tracker, tracks, all_detections):

    score = metrics(tracker)

    win_text = WIN_FONT.render("Complete", 1, WHITE)
    score_text = SCORE_FONT.render(f"Score: {score}", 1, WHITE)

    draw_window(_map, player, destinations, destinations_reached,
                tracker.sensors, tracks, all_detections,
                display_all=True)
    WIN.blit(
        win_text,
        (WIDTH//2 - win_text.get_width()//2,
         HEIGHT//2 - win_text.get_height()//2)
    )
    WIN.blit(
        score_text,
        (WIDTH // 2 - score_text.get_width() // 2,
         3*HEIGHT // 4 - score_text.get_height() // 2)
    )
    pygame.display.update()

    pygame.time.delay(5000)


if __name__ == '__main__':
    main()
