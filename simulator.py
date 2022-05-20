import random

import pygame
import numpy as np
from stonesoup.types.state import State, StateVector, GaussianState
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.movable.movable import FixedMovable
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Euclidean
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.metricgenerator.manager import SimpleManager
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.dataassociator.tracktotrack import TrackToTruth


class Player(pygame.Rect):

    def __init__(self, window_width, window_height, vel, initial_orient, rot_speed, wing_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.win = (window_width, window_height)

        self.vel = vel
        self.orientation = initial_orient
        self.rot_speed = rot_speed

        self.coords = np.array([self.x, self.y], dtype=float)

        self.loc_history = [tuple(self.coords)]

        self.wing_size = wing_size

        self.left_wing_history = [self.left_wingtip()]
        self.right_wing_history = [self.right_wingtip()]
        self.nose_history = [self.nose()]

    @property
    def rotation_vector(self):
        return np.array([self.vel * np.cos(self.orientation),
                         -self.vel * np.sin(self.orientation)])

    def left_wingtip(self):
        arrow_left_x = self.x + self.wing_size * np.cos(
            -self.orientation + np.radians(120))
        arrow_left_y = self.y + self.wing_size * np.sin(
            -self.orientation + np.radians(120))
        return arrow_left_x, arrow_left_y

    def right_wingtip(self):
        arrow_right_x = self.x + self.wing_size * np.cos(
            -self.orientation - np.radians(120))
        arrow_right_y = self.y + self.wing_size * np.sin(
            -self.orientation - np.radians(120))
        return arrow_right_x, arrow_right_y

    def nose(self):
        front_x = self.x + self.wing_size / 2 * np.cos(self.orientation)
        front_y = self.y - self.wing_size / 2 * np.sin(self.orientation)
        return front_x, front_y

    @property
    def next_coords(self):

        next_coords = self.coords + self.rotation_vector

        x, y = next_coords
        if x < 0:
            next_coords = -x, y
            self.orientation = np.radians(180) - self.orientation
        elif x + self.width > self.win[0]:
            next_coords = 2 * self.win[0] - x, y
            self.orientation = np.radians(180) - self.orientation
        x, y = next_coords
        if y < 0:
            next_coords = x, -y
            self.orientation = - self.orientation
        elif y + self.height > self.win[1]:
            next_coords = x, 2 * self.win[1] - y
            self.orientation = - self.orientation
        return next_coords

    def move(self):
        self.coords = self.next_coords
        self.x, self.y = self.coords

        self.update_history()

    def update_history(self):
        self.loc_history.append(tuple(self.coords))
        self.left_wing_history.append(self.left_wingtip())
        self.right_wing_history.append(self.right_wingtip())
        self.nose_history.append(self.nose())


class Missile(Player):
    pass


def make_tracker(sensors_info, detection_period):

    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(0.1),
        ConstantVelocity(0.1)
    ])

    predictor = ExtendedKalmanPredictor(transition_model)

    sensors = set()

    for sensor_info in sensors_info:
        sensor_loc = [sensor_info[0], 0, sensor_info[1], 0]
        sensor_loc = State(StateVector(sensor_loc))

        sensor_range = sensor_info[2]

        sensor = RadarRotatingBearingRange(
            ndim_state=4,
            position_mapping=(0, 2),
            noise_covar=np.diag([np.radians((5*random.random()))**2, 5**2]),
            movement_controller=FixedMovable(states=[sensor_loc],
                                             position_mapping=(0, 2)),
            max_range=sensor_range,
            dwell_center=State([0, 0]),
            rpm=0,
            fov_angle=np.radians(360)
        )
        sensors.add(sensor)

    updater = ExtendedKalmanUpdater()

    deleter = UpdateTimeStepsDeleter(time_steps_since_update=detection_period + 200)

    hypothesiser = DistanceHypothesiser(predictor=predictor,
                                        updater=updater,
                                        measure=Euclidean(),
                                        missed_distance=50)

    data_associator = GNNWith2DAssignment(hypothesiser=hypothesiser)

    prior_state = GaussianState(StateVector([0, 0, 0, 0]),
                                np.diag([500**2, 2**2, 500**2, 2**2]))

    initiator = MultiMeasurementInitiator(prior_state=prior_state,
                                          deleter=deleter,
                                          data_associator=data_associator,
                                          updater=updater,
                                          min_points=2)

    tracker = MultiTargetTracker(initiator=initiator,
                                 deleter=deleter,
                                 detector=None,
                                 data_associator=data_associator,
                                 updater=updater)
    tracker = GameTracker(tracker, sensors)

    return tracker


class GameTracker:

    def __init__(self, tracker, sensors):

        self.tracker = tracker
        self.sensors = sensors

        self.all_detections = list()

        self.groundtruth = GroundTruthPath()
        self.tracks = set()

    def track(self, time, player, detect: bool):

        self.detected = False

        detections_at_time = set()

        player_x, player_y = player.x, player.y
        player_vx = player.vel * np.cos(player.orientation)
        player_vy = player.vel * np.sin(player.orientation)
        player_state = State(StateVector([player_x, player_vx,
                                          player_y, player_vy]),
                             timestamp=time)

        self.groundtruth.append(player_state)

        if detect:
            for sensor in self.sensors:
                detections = sensor.measure({player_state})

                if detections:
                    self.detected = True

                self.tracker.detector = [(time, detections)]
                tracks = next(iter(self.tracker))
                detections_at_time.update(detections)
            self.all_detections.append(detections_at_time)
        else:
            self.tracker.detector = [(time, set())]
            tracks = next(iter(self.tracker))

        self.tracks.update(tracks[1])

        return tracks


def metrics(tracker):

    groundtruth = tracker.groundtruth
    tracks = tracker.tracks

    associator = TrackToTruth(association_threshold=30)

    siap_generator = SIAPMetrics(position_measure=Euclidean((0, 2)),
                                 velocity_measure=Euclidean((1, 3)))

    metric_manager = SimpleManager([siap_generator], associator)

    metric_manager.add_data({groundtruth}, tracks)

    metrics = metric_manager.generate_metrics()

    siap_c = metrics['SIAP Completeness'].value
    siap_c = round(1 - siap_c, 3)

    return siap_c
