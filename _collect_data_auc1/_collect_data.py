# Adapted from:
# https://github.com/carla-simulator/carla/blob/0.8.2/PythonClient/manual_control.py

''''
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
'''

from __future__ import print_function

import argparse
import logging
import random
import time
import os
import sys
import subprocess, signal
import socket
import cv2
import shutil  # For deleting data folder with all subfolders
import csv
from scipy.signal import medfilt

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# To import carla
try:  # Make sure to install CARLA 0.8.2 (prebuilt version)
    carla_path = os.environ['CARLA_PATH']
    sys.path.insert(0, carla_path + 'PythonClient')
except IndexError:
    pass

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
import scipy
import random


# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
out_folder = '/media/cai1/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00/train'
delete_old_data = True  # deletes old out_folder
nVehicles_min = 10
nVehicles_max = 20
nPedestrians_min = 10
nPedestrians_max = 30
town = 'Town01'  # 'Town01' is IL training data, and 'Town02' is testing data
game_fps = 10
episode_time_mins = 10  # 10, in minutes of game time
episodes_count = 120  # 120 train (120*10/60 = 20 hours) & 15 val (15*10/60 = 2.5 hour) (train: 0-->119 old data, 120-->249 new data) (val: 0-->14 old data, 15-->17 new data)
camera_width = 200
camera_height = 88
graphics_mode = 'Epic'  # Low, Medium, High, or Epic

# Steering wheel noise
apply_noise_each_seconds = 1
apply_noise_prob = 0.6  # 0.4
taw_min_seconds = 0.5
taw_max_seconds = 2.0
intensity_min = 0.10
intensity_max = 0.20

# LiDAR OGM (configurations)
horizontal_angle_step = 2  # in degrees, should be divisible by 180
nbrLayers = 32
upper_fov = 10  # in degrees
lower_fov = -30  # in degrees
lidar_range = 150
unreflected_value = float(lidar_range + 10)  # PGM value when no beam received

# High level command generation
radius_squared = 30**2  # 50**2
steer_threshold = 0.06  # in intersections above this threshold means going right or left

show_car_view = False  # False
WINDOW_WIDTH = 800  # car_view window
WINDOW_HEIGHT = 600  # car_view window
show_carla_server_gui = False  # False
kill_running_carla_servers = False  # False
sim_width = 1280
sim_height = 720

fast_mode = False  # False, it's set to False for debugging purpose to run the code faster

# ------------------------------------------------------------------------------
# Constants & initializations
# ------------------------------------------------------------------------------
town1_map_intersect = [(90.923, 0.029), (155.874, 0.029), (336.683, 0.029),
                       (90.923, 57.529), (155.874, 57.529), (336.683, 57.529),
                       (90.923, 130.837), (336.683, 130.837),
                       (90.923, 197.593), (336.683, 197.593),
                       (90.923, 328.542), (336.683, 328.542),
                       ]

inside_noise_drift = False
weathers_dict = {
    0: "Default",
    1: "ClearNoon - Train",  # Train
    2: "CloudyNoon",
    3: "WetNoon - Train",  # Train
    4: "WetCloudyNoon - Test",  # Test
    5: "MidRainyNoon",
    6: "HardRainNoon - Train",  # Train
    7: "SoftRainNoon",
    8: "ClearSunset - Train",  # Train
    9: "CloudySunset",
    10: "WetSunset",
    11: "WetCloudySunset",
    12: "MidRainSunset",
    13: "HardRainSunset",
    14: "SoftRainSunset - Test",  # Test
}

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    weath = random.choice([1, 3, 6, 8])
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=random.randrange(nVehicles_min, nVehicles_max+1),
        NumberOfPedestrians=random.randrange(nPedestrians_min, nPedestrians_max+1),
        WeatherId=weath)
    print("Episode weather: " + weathers_dict[weath])
    settings.randomize_seeds()

    camera_centre = sensor.Camera('CameraRGB_centre')
    camera_centre.set(FOV=100)
    camera_centre.set_image_size(800, 600)
    camera_centre.set_position(2.0, 0.0, 1.4)
    camera_centre.set_rotation(-15.0, 0, 0)
    settings.add_sensor(camera_centre)

    if not fast_mode:
        camera_right = sensor.Camera('CameraRGB_right')
        camera_right.set(FOV=100)
        camera_right.set_image_size(800, 600)
        camera_right.set_position(2.0, 2.0, 1.4)
        camera_right.set_rotation(-15.0, 30.0, 0.0)
        settings.add_sensor(camera_right)

        camera_left = sensor.Camera('CameraRGB_left')
        camera_left.set(FOV=100)
        camera_left.set_image_size(800, 600)
        camera_left.set_position(2.0, -2.0, 1.4)
        camera_left.set_rotation(-15.0, -30.0, 0.0)
        settings.add_sensor(camera_left)

        camera_depth = sensor.Camera('CameraDepth', PostProcessing='Depth')
        camera_depth.set(FOV=100)
        camera_depth.set_image_size(800, 600)
        camera_depth.set_position(2.0, 0.0, 1.4)
        camera_depth.set_rotation(-15.0, 0.0, 0.0)
        settings.add_sensor(camera_depth)

        camera_semseg = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
        camera_semseg.set(FOV=100)
        camera_semseg.set_image_size(800, 600)
        camera_semseg.set_position(2.0, 0.0, 1.4)
        camera_semseg.set_rotation(-15.0, 0.0, 0.0)
        settings.add_sensor(camera_semseg)  

        lidar = sensor.Lidar('Lidar')
        lidar.set_position(0, 0, 2.5)
        lidar.set_rotation(0, 0, 0)
        lidar.set(
            Channels=nbrLayers,
            # 50 is the default (means 50 meter?)
            Range=lidar_range,
            # 100000 is the default, larger values means smaller Horizontal angle per frame
            # For 1 horizontal angle step, points per full rotation = 32 channel * 360/1
            # Points per second = 10*32*360/1 = 115200
            # For 1 horizontal angle: 115200/1 = 115200
            PointsPerSecond=150000,  # game_fps*32*360/1
            # 10 is the default (how many 360 full rotations happen per second). A full rotation is composed of
            # some frames based on FPS (game)
            RotationFrequency=game_fps,
            UpperFovLimit=upper_fov,
            LowerFovLimit=lower_fov)
        settings.add_sensor(lidar)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._image_cam_centre = None
        self._image_cam_right = None
        self._image_cam_left = None
        self._image_cam_depth = None
        self._image_cam_semsegm = None
        self._lidar_image = None
        self._lidar_pgm_image = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None  # non-player agents
        self.episodes_done = 0

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                # Commented in case the script is called with nohup ... &
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         return
                self._on_loop()
                if show_car_view:
                    self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if show_car_view:
            if self._city_name is not None:
                self._display = pygame.display.set_mode(
                    (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)
            else:
                self._display = pygame.display.set_mode(
                    (WINDOW_WIDTH, WINDOW_HEIGHT),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        weath = random.choice([1, 3, 6, 8])
        self._carla_settings.set(SynchronousMode=True,
                                 SendNonPlayerAgentsInfo=True,
                                 NumberOfVehicles=random.randrange(nVehicles_min, nVehicles_max + 1),
                                 NumberOfPedestrians=random.randrange(nPedestrians_min, nPedestrians_max + 1),
                                 WeatherId=weath)
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        print("Episode weather: " + weathers_dict[weath])
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

        # Prepare folders to save collected data
        print("Creating a new episode folder inside out_folder!")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        tmpDir = out_folder
        cnt = 0
        while os.path.isdir(tmpDir):
            tmpDir = out_folder + "/%i" % cnt
            cnt += 1
        self.out_folder_episode = tmpDir
        os.makedirs(self.out_folder_episode)
        os.makedirs(self.out_folder_episode + '/CameraRGB_centre')
        os.makedirs(self.out_folder_episode + '/CameraRGB_right')
        os.makedirs(self.out_folder_episode + '/CameraRGB_left')
        os.makedirs(self.out_folder_episode + '/CameraDepth')
        os.makedirs(self.out_folder_episode + '/CameraSemSeg')
        os.makedirs(self.out_folder_episode + '/LiDAR_Topview')
        os.makedirs(self.out_folder_episode + '/LiDAR_PGM')

        # metadata file
        self.metadata_csv_file = open(self.out_folder_episode + '/metadata.csv', 'w+')
        self.metadata_csv_writer = csv.writer(self.metadata_csv_file, delimiter=',')
        self.metadata_csv_writer.writerow(
            ['step', 'steer', 'throttle', 'brake', 'forward_speed', 'yaw', 'location_x', 'location_y'])

    # ------------------------------------------------------------------------------
    # Cartesian to Polar coordinates conversion function
    # ------------------------------------------------------------------------------
    @staticmethod
    def cart2pol(x, y, z):
        xy = x ** 2 + y ** 2
        rho = np.sqrt(xy + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(xy))  # np.arctan2 retruns from -np.pi to np.pi

        # make angles from 0 to 360
        theta_deg = (np.degrees(theta) + 360) % 360
        phi_deg = (np.degrees(phi) + 360) % 360

        return rho, theta_deg, phi_deg

    # ------------------------------------------------------------------------------
    # When episode ends create the high level commands
    # ------------------------------------------------------------------------------
    @staticmethod
    def _on_episode_end(input_csv_file):
        # read input file
        file_lines = []
        x_list = []
        y_list = []
        steer_list = []
        speeds_list = []
        with open(input_csv_file, encoding='utf-8') as csvfile:
            readfile = csv.reader(csvfile, delimiter=',')
            for row in readfile:
                if row[0] == 'step':  # ignore header
                    continue
                file_lines.append(row)
                x_list.append(float(row[6]))
                y_list.append(float(row[7]))
                steer_list.append(float(row[1]))
                speeds_list.append(float(row[4]))

        # Create high_lvl_cmd_list (intial one that is useless; will be updated by calling _preprocess_highlvlcmd)
        high_lvl_cmd_list = []
        output_csv_file = open(input_csv_file.replace('metadata.csv', 'metadata_with_highlvlcmd.csv'), 'w+')
        output_csv_file_writer = csv.writer(output_csv_file, delimiter=',')
        output_csv_file_writer.writerow(
            ['step', 'steer', 'throttle', 'brake', 'forward_speed', 'yaw', 'location_x', 'location_y', 'high_lvl_cmd'])
        # high_lvl_cmds: 0 Right, 1 Straight, 2 Follow lane, 3 Left
        for i in range(len(x_list)):
            cmd = 2  # Follow Lane
            # If within intersection and brake is not zero
            if speeds_list[i] > 0:
                for j in range(len(town1_map_intersect)):
                    if ((x_list[i] - town1_map_intersect[j][0]) ** 2 + (y_list[i] - town1_map_intersect[j][1]) ** 2) < \
                            radius_squared:
                        if float(steer_list[i]) > steer_threshold and float(steer_list[i]) != 0.0:
                            cmd = 0  # Right
                        elif float(steer_list[i]) < -steer_threshold and float(steer_list[i]) != 0.0:
                            cmd = 3  # Left
                        elif float(steer_list[i]) != 0.0:
                            cmd = 1  # Straight
                        break
            high_lvl_cmd_list.append(cmd)

        # Smooth the high level command
        high_lvl_cmd_list = medfilt([int(i) for i in high_lvl_cmd_list], 75)  # window size should be odd
        high_lvl_cmd_list = [int(i) for i in high_lvl_cmd_list]

        # Add high_lvl_cmd_list information to metadata CSV file
        for i in range(len(high_lvl_cmd_list)):
            file_lines[i].append(high_lvl_cmd_list[i])
        for i in range(len(file_lines)):
            output_csv_file_writer.writerow(file_lines[i])
            output_csv_file.flush()

    def no_car_around(self, measurements):
        for agent in self._agent_positions:  # Other cars
            if agent.HasField('vehicle'):
                other_car = self._map.convert_to_pixel([agent.vehicle.transform.location.x,
                                                        agent.vehicle.transform.location.y,
                                                        agent.vehicle.transform.location.z])
                our_car = self._map.convert_to_pixel([measurements.player_measurements.transform.location.x,
                                                      measurements.player_measurements.transform.location.y,
                                                      measurements.player_measurements.transform.location.z])
                if np.sqrt((other_car[1]-our_car[1])**2+(other_car[0]-our_car[0])**2) < 50:
                    return False

        return True

    def _on_loop(self):
        global inside_noise_drift
        global t_start, taw, intensity, sign
        self._timer.tick()

        measurements, sensor_data = self.client.read_data()

        self._image_cam_centre = image_converter.to_rgb_array(sensor_data.get('CameraRGB_centre', None))
        if not fast_mode:
            self._image_cam_right = image_converter.to_rgb_array(sensor_data.get('CameraRGB_right', None))
            self._image_cam_left = image_converter.to_rgb_array(sensor_data.get('CameraRGB_left', None))
            self._image_cam_depth = image_converter.depth_to_logarithmic_grayscale(sensor_data.get('CameraDepth', None))
            self._image_cam_semsegm = image_converter.labels_to_cityscapes_palette(sensor_data.get('CameraSemSeg', None))
            self._lidar_measurement = sensor_data.get('Lidar', None)

        # resize images
        self._image_cam_centre = scipy.misc.imresize(self._image_cam_centre, [camera_height, camera_width])
        if not fast_mode:
            self._image_cam_right = scipy.misc.imresize(self._image_cam_right, [camera_height, camera_width])
            self._image_cam_left = scipy.misc.imresize(self._image_cam_left, [camera_height, camera_width])
            self._image_cam_depth = scipy.misc.imresize(self._image_cam_depth, [camera_height, camera_width])
            self._image_cam_semsegm = scipy.misc.imresize(self._image_cam_semsegm, [camera_height, camera_width])

            # LiDAR top view image
            zoomout_factor = 3.0
            lidar_data = np.array(self._lidar_measurement.data[:, :2])/zoomout_factor
            lidar_data *= 2.0
            lidar_data += 100.0
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            # draw lidar
            lidar_img_size = (200, 200, 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._lidar_image = lidar_img.swapaxes(0, 1)

            # LiDAR PGM
            lidar_data = np.array(self._lidar_measurement.data)
            x = lidar_data[:, 0]
            y = lidar_data[:, 1]
            z = -lidar_data[:, 2]
            (rho, theta_deg, phi_deg) = CarlaGame.cart2pol(x, y, z)  # theta of 90 means front-facing
            unique_phis = (np.arange(upper_fov, lower_fov, -(upper_fov - lower_fov)/nbrLayers) -
                           (upper_fov - lower_fov)/(2.*nbrLayers) + 360) % 360
            unique_thetas = np.arange(180, 360, horizontal_angle_step) + horizontal_angle_step/2.  # only range front-facing
            self._lidar_pgm_image = np.ones((len(unique_phis), len(unique_thetas))) * unreflected_value
            for i in range(self._lidar_pgm_image.shape[0]):  # For each layer
                for j in range(self._lidar_pgm_image.shape[1]):  # For each group of neighboring beams
                    # TODO: I found that multiplying with 1.1 (a number slightly bigger than 1) prevents PGM artifacts, why?
                    indices_phi = np.abs(phi_deg - unique_phis[i]) <= 1.1*(upper_fov - lower_fov)/(2.*nbrLayers)
                    indices_theta = np.abs(theta_deg - unique_thetas[j]) <= horizontal_angle_step/2.
                    rhos = rho[indices_phi & indices_theta]
                    if len(rhos) > 0:
                        self._lidar_pgm_image[i, j] = np.mean(rhos)
            self._lidar_pgm_image = 255.0 * np.repeat(self._lidar_pgm_image[:, :, np.newaxis], 3, axis=2) / \
                                    unreflected_value

        # Print measurements every some time
        if self._timer.step % 1 == 0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(measurements.player_measurements, map_position, lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)

            self._timer.lap()

        if not fast_mode:
            # Save data in BGR format, later open CV when reading it read it as BGR
            cv2.imwrite(self.out_folder_episode + '/CameraRGB_centre/%08d.png' % self._timer.step, self._image_cam_centre)
            cv2.imwrite(self.out_folder_episode + '/CameraRGB_right/%08d.png' % self._timer.step, self._image_cam_right)
            cv2.imwrite(self.out_folder_episode + '/CameraRGB_left/%08d.png' % self._timer.step, self._image_cam_left)
            cv2.imwrite(self.out_folder_episode + '/CameraDepth/%08d.png' % self._timer.step, self._image_cam_depth)
            cv2.imwrite(self.out_folder_episode + '/CameraSemSeg/%08d.png' % self._timer.step, self._image_cam_semsegm)
            cv2.imwrite(self.out_folder_episode + '/LiDAR_Topview/%08d.png' % self._timer.step, self._lidar_image)
            cv2.imwrite(self.out_folder_episode + '/LiDAR_PGM/%08d.png' % self._timer.step, self._lidar_pgm_image)

            # Save metadata
            self.metadata_csv_writer.writerow([self._timer.step,
                                               measurements.player_measurements.autopilot_control.steer,
                                               measurements.player_measurements.autopilot_control.throttle,
                                               measurements.player_measurements.autopilot_control.brake,
                                               measurements.player_measurements.forward_speed,
                                               measurements.player_measurements.transform.rotation.yaw,
                                               measurements.player_measurements.transform.location.x,
                                               measurements.player_measurements.transform.location.y,
                                               ])
            self.metadata_csv_file.flush()
            self.egocar_x = measurements.player_measurements.transform.location.x
            self.egocar_y = measurements.player_measurements.transform.location.y

        # Control
        control = None
        if show_car_view:
            control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if (show_car_view and control is None) or (self._timer.step == game_fps*60*episode_time_mins):  # episode ended
            self._on_episode_end(self.out_folder_episode + '/metadata.csv')
            self.episodes_done += 1
            if self.episodes_done == episodes_count:
                print("Data Collection Finished!")
                exit()
            inside_noise_drift = False
            self._on_new_episode()
        # Collision happens
        elif (measurements.player_measurements.collision_vehicles > 0) or \
                (measurements.player_measurements.collision_pedestrians > 0) or \
                (measurements.player_measurements.collision_other > 0):
            print("Collision happens, deleting episdoe data and restarting!")
            shutil.rmtree(self.out_folder_episode, ignore_errors=True)
            inside_noise_drift = False
            self._on_new_episode()
        elif self._enable_autopilot:
            ctrl = measurements.player_measurements.autopilot_control
            # Add noise
            if (not inside_noise_drift) and (self._timer.step % (game_fps * apply_noise_each_seconds) == 0) and \
                    (random.random() <= apply_noise_prob):
                inside_noise_drift = True
                t_start = self._timer.step
                taw = random.randrange(int(game_fps*taw_min_seconds), int(game_fps*taw_max_seconds)+1)
                intensity = random.uniform(intensity_min, intensity_max)
                sign = 1 if random.random() < 0.5 else -1
            if inside_noise_drift:
                noise = sign * intensity * max(0, 1-abs(2*(self._timer.step-t_start)/taw-1))
                ctrl.steer += noise
                if self._timer.step >= (t_start + taw):
                    inside_noise_drift = False
                print("  %g noise is applied, t_start=%d, taw=%d, intensity=%g" %
                      (noise, t_start, taw, sign*intensity))

            # Don't drive very fast
            if measurements.player_measurements.forward_speed >= 20:
                ctrl.throttle /= 2.0

            # Avoid stopping? We already remove episode if collision happens  TODO: caused deadlock and crashes
            '''if (measurements.player_measurements.forward_speed <= 3.0) and self.no_car_around(measurements):
                ctrl.throttle = 1.0
                ctrl.brake = 0.0'''

            self.client.send_control(ctrl)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        # print_over_same_line(message)
        print(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        # print_over_same_line(message)
        print(message)

    def _on_render(self):
        if self._image_cam_left is not None:
            surface = pygame.surfarray.make_surface(self._image_cam_left.swapaxes(0, 1))
            self._display.blit(surface, (10, 10))

        if self._image_cam_centre is not None:
            surface = pygame.surfarray.make_surface(self._image_cam_centre.swapaxes(0, 1))
            self._display.blit(surface, (220, 10))

        if self._image_cam_right is not None:
            surface = pygame.surfarray.make_surface(self._image_cam_right.swapaxes(0, 1))
            self._display.blit(surface, (430, 10))

        if self._image_cam_depth is not None:
            surface = pygame.surfarray.make_surface(self._image_cam_depth.swapaxes(0, 1))
            self._display.blit(surface, (10, 130))

        if self._image_cam_semsegm is not None:
            surface = pygame.surfarray.make_surface(self._image_cam_semsegm.swapaxes(0, 1))
            self._display.blit(surface, (220, 130))

        if self._lidar_measurement is not None:
            surface = pygame.surfarray.make_surface(self._lidar_pgm_image.swapaxes(0, 1))
            zoomin_factor = 4
            surface = pygame.transform.scale(surface,
                                             (self._lidar_pgm_image.shape[1] * zoomin_factor,
                                              self._lidar_pgm_image.shape[0] * zoomin_factor))
            pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(0, 0, self._lidar_pgm_image.shape[1] * zoomin_factor,
                                                                   self._lidar_pgm_image.shape[0] * zoomin_factor), 10)
            self._display.blit(surface, (10, 260))

            surface = pygame.surfarray.make_surface(self._lidar_image.swapaxes(0, 1))
            pygame.draw.rect(surface, (255, 255, 255), pygame.Rect(0, 0, self._lidar_image.shape[1],
                                                                   self._lidar_image.shape[0]), 10)
            self._display.blit(surface, (430, 260))

        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]
            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            intersection = False
            for j in range(len(town1_map_intersect)):
                intersection_position = self._map.convert_to_pixel(
                    [town1_map_intersect[j][0], town1_map_intersect[j][1], 0])
                w_pos = int(intersection_position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
                h_pos = int(intersection_position[1] * (new_window_width / float(self._map_shape[1])))
                pygame.draw.circle(surface, [0, 0, 255, 255], (w_pos, h_pos), 4, 0)
                pygame.draw.circle(surface, (0, 0, 255, 100), (w_pos, h_pos), 2*int(np.sqrt(radius_squared)), 10)
                # Record if car is within intersection
                if ((self.egocar_x - town1_map_intersect[j][0]) ** 2 +
                    (self.egocar_y - town1_map_intersect[j][1]) ** 2) < radius_squared:
                    intersection = True
            if intersection:
                print_over_same_line('Car within intersection: Yes')
            else:
                print_over_same_line('Car within intersection: No')

            # Ego-car
            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))
            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)

            for agent in self._agent_positions:  # Other cars
                if agent.HasField('vehicle'):
                    agent_position = self._map.convert_to_pixel([
                        agent.vehicle.transform.location.x,
                        agent.vehicle.transform.location.y,
                        agent.vehicle.transform.location.z])
                    w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
                    h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

                    pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

            self._display.blit(surface, (WINDOW_WIDTH, 0))

        pygame.display.flip()


def main():
    # Get port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()

    # argparser
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=port,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_false',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=town,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    # Kill the server if already running
    if kill_running_carla_servers:
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'CarlaUE4' in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)

    # Delete out folder
    if delete_old_data:
        print("Deleting old out_folder content!")
        shutil.rmtree(out_folder, ignore_errors=True)

    # Run Carla server
    bashCommand = carla_path + 'CarlaUE4.sh /Game/Maps/' + args.map_name + \
                  ' -windowed -ResX=' + str(sim_width) + ' -ResY=' + str(sim_height) + ' -world-port=' + \
                  str(args.port) + ' -carla-server-timeout=10000000ms' + \
                  ' -carla-server -benchmark -fps=' + str(game_fps) + \
                  ' -quality-level=' + graphics_mode

    # bashCommand = 'vglrun -d :7.' + str(gpu_id) + ' ' + bashCommand
    # If headless start of the simulator (no real window; headless)
    if (not show_carla_server_gui):
        # setting the environment variable DISPLAY to empty
        bashCommand = 'DISPLAY= ' + bashCommand
    print("CARLA server command called: " + bashCommand)
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(bashCommand, shell=True, stdout=FNULL, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
    # output, error = process.communicate()
    # print('STDOUT:{}'.format(output))
    time.sleep(10.0)  # Wait until CARLA is initialized

    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
