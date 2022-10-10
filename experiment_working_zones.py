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

    """
    Town 1 navigation benchmark compared to CORL17 benchmark:
        experiments require reroute: 12
        experiments require no reroute: 13
        experiments changed: 9
    1-  [105, 29], Reroute
    2-  [27, 130], No Reroute
    3-  [102, 87], Reroute
    4-  [132, 27], Reroute
    5-  [24, 44], No Reroute
    6-  [96, 26], No Reroute
    7-  [34, 67], No Reroute
    8-  [28, 1], No WZ met --> replace by [54, 99], No Reroute
    9-  [140, 134], Deadlock --> replace by [140, 56], Reroute
    10- [105, 9], No Reroute
    11- [148, 129], No Reroute
    12- [65, 18], Deadlock --> replace by [45, 23], Reroute
    13- [21, 16], Deadlock --> replace by [21, 20], Reroute
    14- [147, 97], No Reroute
    15- [42, 51], No Reroute
    16- [30, 41], No Reroute
    17- [18, 107], on WZ --> replace by [20, 107], No Reroute
    18- [69, 45], No WZ met --> replace by [93, 129], Reroute
    19- [102, 95], Reroute
    20- [18, 145], on WZ and Deadlock--> replace by [16, 67], No Reroute
    21- [111, 64], Reroute
    22- [79, 45], No WZ met --> replace by [32, 77], Reroute
    23- [84, 69], No Reroute
    24- [73, 31], No WZ met --> replace by [32, 149], Reroute
    25- [37, 81], Reroute

    Town 2 navigation benchmark compared to CORL17 benchmark:
        experiments require reroute: 13
        experiments require no reroute: 12
        experiments changed: 7
    1-  [19, 66] Reroute
    2-  [79, 14]  Reroute
    3-  [19, 57] No Reroute
    4-  [23, 1] No WZ met --> replace by [14, 31], Reroute
    5-  [53, 76]  Reroute
    6-  [42, 13] No Reroute
    7-  [31, 71] Reroute
    8-  [33, 5]  No WZ met --> replace by [77, 41], Reroute
    9-  [54, 30] Deadlock --> replace by [13, 23], No Reroute
    10- [10, 61] No Reroute
    11- [66, 3] Deadlock --> replace by [27, 7], Reroute
    12- [27, 12] Reroute
    13- [79, 19] Reroute
    14- [2, 29] No Reroute
    15- [16, 14]  Reroute
    16- [5, 57] No Reroute
    17- [70, 73] Reroute
    18- [46, 67] No Reroute
    19- [57, 50] No WZ met --> replace by [9, 79], No Reroute
    20- [61, 49] No Reroute
    21- [21, 12] No WZ met --> replace by [16, 42], Reroute
    22- [51, 81] No Reroute
    23- [77, 68] No Reroute
    24- [56, 65] No WZ met --> replace by [16, 19], Reroute
    25- [43, 54] No Reroute"""

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks

        """

        def _poses_navigation():
            return [[105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
                    [96, 26], [34, 67], [54, 99], [140, 56], [105, 9],
                    [148, 129], [45, 23], [21, 20], [147, 97], [42, 51],
                    [30, 41], [20, 107], [93, 129], [102, 95], [16, 67],
                    [111, 64], [32, 77], [84, 69], [32, 149], [37, 81]]

        return [_poses_navigation(),
                _poses_navigation()]

    def _poses_town02(self):

        def _poses_navigation():
            return [[19, 66], [79, 14], [19, 57], [14, 31],
                    [53, 76], [42, 13], [31, 71], [77, 41],
                    [13, 23], [10, 61], [27, 7], [27, 12],
                    [79, 19], [2, 29], [16, 14], [5, 57],
                    [70, 73], [46, 67], [9, 79], [61, 49], [16, 42],
                    [51, 81], [77, 68], [16, 19], [43, 54]]

        return [_poses_navigation(),
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
            vehicles_tasks = [0, 20]
            pedestrians_tasks = [0, 50]
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0, 15]
            pedestrians_tasks = [0, 50]

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
