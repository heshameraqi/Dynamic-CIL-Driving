# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function

from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite


class experiment(ExperimentSuite):
    def __init__(self, model_inputs_mode, our_camera_setup, city_name):
        self.model_inputs_mode = model_inputs_mode
        self.our_camera_setup = our_camera_setup
        super().__init__(city_name)

    #  {1: 'Clear Noon', 3: 'After Rain Noon',  6: 'Heavy Rain Noon', 8: 'Clear Sunset', 4: 'Cloudy After Rain', 14: 'Soft Rain Sunset'}
    @property
    def train_weathers(self):
        # return [6] * 4
        return [1, 3, 6, 8]

    @property
    def test_weathers(self):
        # return [6] * 2
        return [4, 14]

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks

        """

        def _poses_straight():
            return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
                    [68, 50], [61, 59], [47, 64], [147, 90], [33, 87],
                    [26, 19], [80, 76], [45, 49], [55, 44], [29, 107],
                    [95, 104], [84, 34], [53, 67], [22, 17], [91, 148],
                    [20, 107], [78, 70], [95, 102], [68, 44], [45, 69]]

        def _poses_one_curve():
            return [[138, 17], [47, 16], [26, 9], [42, 49], [140, 124],
                    [85, 98], [65, 133], [137, 51], [76, 66], [46, 39],
                    [40, 60], [0, 29], [4, 129], [121, 140], [2, 129],
                    [78, 44], [68, 85], [41, 102], [95, 70], [68, 129],
                    [84, 69], [47, 79], [110, 15], [130, 17], [0, 17]]

        def _poses_navigation():
            return [[105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
                    [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
                    [148, 129], [65, 18], [21, 16], [147, 97], [42, 51],
                    [30, 41], [18, 107], [69, 45], [102, 95], [18, 145],
                    [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]

        return [_poses_straight(),
                _poses_one_curve(),
                _poses_navigation(),
                _poses_navigation()]

    def _poses_town02(self):

        def _poses_straight():
            return [[38, 34], [4, 2], [12, 10], [62, 55], [43, 47],
                    [64, 66], [78, 76], [59, 57], [61, 18], [35, 39],
                    [12, 8], [0, 18], [75, 68], [54, 60], [45, 49],
                    [46, 42], [53, 46], [80, 29], [65, 63], [0, 81],
                    [54, 63], [51, 42], [16, 19], [17, 26], [77, 68]]

        def _poses_one_curve():
            return [[37, 76], [8, 24], [60, 69], [38, 10], [21, 1],
                    [58, 71], [74, 32], [44, 0], [71, 16], [14, 24],
                    [34, 11], [43, 14], [75, 16], [80, 21], [3, 23],
                    [75, 59], [50, 47], [11, 19], [77, 34], [79, 25],
                    [40, 63], [58, 76], [79, 55], [16, 61], [27, 11]]

        def _poses_navigation():
            return [[19, 66], [79, 14], [19, 57], [23, 1],
                    [53, 76], [42, 13], [31, 71], [33, 5],
                    [54, 30], [10, 61], [66, 3], [27, 12],
                    [79, 19], [2, 29], [16, 14], [5, 57],
                    [70, 73], [46, 67], [57, 50], [61, 49], [21, 12],
                    [51, 81], [77, 68], [56, 65], [43, 54]]

        return [_poses_straight(),
                _poses_one_curve(),
                _poses_navigation(),
                _poses_navigation()
                ]

    def build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.


        """

        # We set the Sensors
        # This single RGB camera is used on every experiment
        # Sensor configs TODO: should be the same as during data collection automatically (_collect_data.py)
        game_fps = 15  # TODO: Should also be the same as in _deploy.py
        camera_width = 200
        camera_height = 88
        nbrLayers = 32
        upper_fov = 10  # in degrees
        lower_fov = -30  # in degrees
        lidar_range = 50

        camera_centre = Camera('CameraRGB_centre')
        camera_centre.set(FOV=100)
        camera_centre.set_image_size(800, 600)
        camera_centre.set_position(2.0, 0.0, 1.4)
        camera_centre.set_rotation(-15.0, 0, 0)

        if self.model_inputs_mode == "3cams" or self.model_inputs_mode == "3cams-pgm":
            camera_right = Camera('CameraRGB_right')
            camera_right.set(FOV=100)
            camera_right.set_image_size(800, 600)
            camera_right.set_position(2.0, 0.0, 1.4)
            camera_right.set_rotation(-15.0, 30.0, 0.0)

            camera_left = Camera('CameraRGB_left')
            camera_left.set(FOV=100)
            camera_left.set_image_size(800, 600)
            camera_left.set_position(2.0, 0.0, 1.4)
            camera_left.set_rotation(-15.0, -30.0, 0.0)

        if self.model_inputs_mode == "3cams-pgm" or self.model_inputs_mode == "1cam-pgm":
            lidar = Lidar('Lidar')
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
                PointsPerSecond=150000,  # game_fps * 32 * 360 / 1
                # 10 is the default (how many 360 full rotations happen per second). A full rotation is composed of
                # some frames based on FPS (game)
                RotationFrequency=game_fps,
                UpperFovLimit=upper_fov,
                LowerFovLimit=lower_fov)

            lidar_long_range = Lidar('Lidar_ogm_long_range')
            lidar_long_range.set_position(0, 0.0, 2.5)  # should be as semantic segm camera
            lidar_long_range.set_rotation(0, 0,
                                          0)  # should be as semantic segm camera except mid element represnting yaw rotation
            lidar_long_range.set(
                Channels=32,
                # 50 is the default (means 50 meter?)
                Range=100,
                # 100000 is the default, larger values means smaller Horizontal angle per frame
                # For 1 horizontal angle step, points per full rotation = 32 channel * 360/1
                # Points per second = 10*32*360/1 = 115200
                # For 1 horizontal angle: 115200/1 = 115200
                PointsPerSecond=150000,  # game_fps * 32 * 360 / 1
                # 10 is the default (how many 360 full rotations happen per second). A full rotation is composed of
                # some frames based on FPS (game)
                RotationFrequency=game_fps,
                UpperFovLimit=upper_fov,
                LowerFovLimit=lower_fov)
            """lidar_long_range.set(
                    Channels=4,
                    # 50 is the default (means 50 meter?)
                    Range=100,
                    # 100000 is the default, larger values means smaller Horizontal angle per frame
                    # For 1 horizontal angle step, points per full rotation = 32 channel * 360/1
                    # Points per second = 10*32*360/1 = 115200
                    # For 1 horizontal angle: 115200/1 = 115200
                    PointsPerSecond=20000,  # game_fps * 32 * 360 / 1
                    # 10 is the default (how many 360 full rotations happen per second). A full rotation is composed of
                    # some frames based on FPS (game)
                    RotationFrequency=game_fps,
                    UpperFovLimit=0.8,
                    LowerFovLimit=-1.6)"""

            camera_semseg = Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
            camera_semseg.set(FOV=100)
            camera_semseg.set_image_size(800, 600)
            camera_semseg.set_position(0, 0.0, 2.5)
            camera_semseg.set_rotation(0, 0.0, 0.0)

        # Number of vehicles and pedestrians
        if self._city_name == 'Town01':
            poses_tasks = self._poses_town01()
            vehicles_tasks = [0, 0, 0, 20]
            pedestrians_tasks = [0, 0, 0, 50]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0, 0, 0, 15]
            pedestrians_tasks = [0, 0, 0, 50]

        experiments_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera_centre)
                if self.our_camera_setup:
                    if self.model_inputs_mode == "3cams" or self.model_inputs_mode == "3cams-pgm":
                        conditions.add_sensor(camera_right)
                        conditions.add_sensor(camera_left)
                    if self.model_inputs_mode == "3cams-pgm" or self.model_inputs_mode == "1cam-pgm":
                        conditions.add_sensor(lidar)
                        conditions.add_sensor(lidar_long_range)
                        conditions.add_sensor(camera_semseg)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector
