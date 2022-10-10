# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import abc
import logging
import math
import time
from time import gmtime, strftime
import sys

from carla.client import VehicleControl
from carla.client import make_carla_client
from carla.driving_benchmark.metrics import Metrics
from planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from carla.driving_benchmark import results_printer
from recording import Recording

# from ogm_planner import OGM_Planner
from ogm_planner import OGM_Planner

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class DrivingBenchmark(object):
    """
    The Benchmark class, controls the execution of the benchmark interfacing
    an Agent class with a set Suite.


    The benchmark class must be inherited with a class that defines the
    all the experiments to be run by the agent
    """

    def __init__(
            self,
            city_name='Town01',
            name_to_save='Test',
            continue_experiment=False,
            start_from_exp_pose=(1,1),
            save_images=False,
            distance_for_success=2.0,
            enable_WZ_avoidance_using_OGM=False,
            enable_steering_rect_using_OGM=False,
            simulator_fps=15,
            visualize_ogm_planner=False,
            save_ogm_planner_figure=False,
            start_visualize_or_save_from_frame=0,
            normal_save_quality=50,
            save_high_quality_frame_numbers=[],
            visualize_save_directory=""
    ):

        self.__metaclass__ = abc.ABCMeta

        self.COMMANDS_ENUM = {
            REACH_GOAL: "REACH_GOAL",
            GO_STRAIGHT: "GO_STRAIGHT",
            TURN_RIGHT: "TURN_RIGHT",
            TURN_LEFT: "TURN_LEFT",
            LANE_FOLLOW: "LANE_FOLLOW",
        }

        self._city_name = city_name
        self._base_name = name_to_save
        # The minimum distance for arriving into the goal point in
        # order to consider ir a success
        self._distance_for_success = distance_for_success
        # The object used to record the benchmark and to able to continue after
        self._recording = Recording(name_to_save=name_to_save,
                                    continue_experiment=continue_experiment,
                                    save_images=save_images,
                                    start_from_exp_pose=start_from_exp_pose
                                    )

        # We have a default planner instantiated that produces high level commands
        self._planner = Planner(city_name)
        self._enable_WZ_avoidance_using_OGM = enable_WZ_avoidance_using_OGM
        self._enable_steering_rect_using_OGM = enable_steering_rect_using_OGM
        if self._enable_WZ_avoidance_using_OGM:
            self._ogm_planner = OGM_Planner(self._city_name, simulator_fps=simulator_fps, route_planner=self._planner,
                                            visualize_ogm_planner=visualize_ogm_planner,
                                            save_ogm_planner_figure = save_ogm_planner_figure,
                                            start_visualize_or_save_from_frame=start_visualize_or_save_from_frame,
                                            normal_save_quality=normal_save_quality,
                                            save_high_quality_frame_numbers=save_high_quality_frame_numbers,
                                            visualize_save_directory=visualize_save_directory)

    def benchmark_agent(self, experiment_suite, agent, client):
        """
        Function to benchmark the agent.
        It first check the log file for this benchmark.
        if it exist it continues from the experiment where it stopped.
        Args:
            experiment_suite
            agent: an agent object with the run step class implemented.
            client:
        Return:
            A dictionary with all the metrics computed from the
            agent running the set of experiments.
        """

        # Instantiate a metric object that will be used to compute the metrics for
        # the benchmark afterwards.
        metrics_object = Metrics(experiment_suite.metrics_parameters,
                                 experiment_suite.dynamic_tasks)

        # Function return the current pose and task for this benchmark.
        start_pose, start_experiment = self._recording.get_pose_and_experiment(
            experiment_suite.get_number_of_poses_task())
        # if sys.platform.startswith('win'):  # Windows machines workaround
        #     start_pose -= 1

        logging.info('START')
        ex = 0
        exs = len(experiment_suite.get_experiments()[int(start_experiment):])
        for experiment in experiment_suite.get_experiments()[int(start_experiment):]:
            ex += 1
            positions = client.load_settings(
                experiment.conditions).player_start_spots

            self._recording.log_start(experiment.task)

            pos = 0
            poss = len(experiment.poses[start_pose:])
            for pose in experiment.poses[start_pose:]:
                pos += 1
                for rep in range(experiment.repetitions):
                    start_index = pose[0]
                    end_index = pose[1]

                    if self._enable_WZ_avoidance_using_OGM:
                        self._ogm_planner.set_source_destination_from_GPS(positions[start_index], positions[end_index])

                    client.start_episode(start_index)
                    # Print information on
                    timenow = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                    logging.info('\n======== Time: ' + timenow)
                    logging.info('\n======== Experiment: %i/%i - Pose: %i/%i - Repetition: %i/%i ...' %
                                 (ex, exs, pos, poss, rep + 1, experiment.repetitions))
                    logging.info(' Start Position %d End Position %d ',
                                 start_index, end_index)

                    self._recording.log_poses(start_index, end_index,
                                              experiment.Conditions.WeatherId)

                    # Calculate the initial distance for this episode
                    initial_distance = \
                        sldist(
                            [positions[start_index].location.x, positions[start_index].location.y],
                            [positions[end_index].location.x, positions[end_index].location.y])

                    # Calculate experiment timeout
                    time_out = experiment_suite.calculate_time_out(
                        self._get_shortest_path(positions[start_index], positions[end_index]))

                    # running the agent
                    (result, reward_vec, control_vec, final_time, remaining_distance) = \
                        self._run_navigation_episode(
                            agent, client, time_out, positions[end_index],
                            str(experiment.Conditions.WeatherId) + '_'
                            + str(experiment.task) + '_' + str(start_index)
                            + '.' + str(end_index))

                    # Write the general status of the just ran episode
                    self._recording.write_summary_results(
                        experiment, pose, rep, initial_distance,
                        remaining_distance, final_time, time_out, result)

                    # Write the details of this episode.
                    self._recording.write_measurements_results(experiment, rep, pose, reward_vec,
                                                               control_vec)
                    if result > 0:
                        logging.info('+++++ Target achieved in %f seconds! +++++',
                                     final_time)
                    else:
                        logging.info('----- Timeout! -----')

            start_pose = 0

        self._recording.log_end()

        return metrics_object.compute(self._recording.path)

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _get_shortest_path(self, start_point, end_point):
        """
        Calculates the shortest path between two points considering the road netowrk
        """

        return self._planner.get_shortest_path_distance(
            [
                start_point.location.x, start_point.location.y, 0.22], [
                start_point.orientation.x, start_point.orientation.y, 0.22], [
                end_point.location.x, end_point.location.y, end_point.location.z], [
                end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    def _run_navigation_episode(
            self,
            agent,
            client,
            time_out,
            target,
            episode_name):
        """
         Run one episode of the benchmark (Pose) for a certain agent.
        Args:
            agent: the agent object
            client: an object of the carla client to communicate
            with the CARLA simulator
            time_out: the time limit to complete this episode
            target: the target position to reach
            episode_name: The name for saving images of this episode
        """

        # Send an initial command.
        measurements, sensor_data = client.read_data()
        client.send_control(VehicleControl())

        initial_timestamp = measurements.game_timestamp
        current_timestamp = initial_timestamp

        # The vector containing all measurements produced on this episode
        measurement_vec = []
        # The vector containing all controls produced on this episode
        control_vec = []
        frame = 0
        distance = 10000
        success = False

        # Reset planner
        self._planner._previous_node = None

        # time_out is in seconds
        time_out = time_out * 1.3  # Increased timeout due to possible planner rerouting due to OGM occupancy due to something like Working Zones  #TODO: is 1.3 scale enough?
        while (current_timestamp - initial_timestamp) < ((time_out + 0) * 1000) and not success:

            # Read data from server with the client
            measurements, sensor_data = client.read_data()

            # The directions to reach the goal are calculated (Planner get_next_command)
            directions = self._get_directions(measurements.player_measurements.transform, target)

            # Agent process the data. Modified to return speed
            control, speed, lidar_pgm_image = agent.run_step(measurements, sensor_data, directions, target)

            # OGM handling
            # directions: {0: 'REACH_GOAL', 2: 'LANE_FOLLOW', 3: 'TURN_LEFT', 4: 'TURN_RIGHT', 5: 'GO_STRAIGHT'}
            if self._enable_WZ_avoidance_using_OGM:  # and measurements.frame_number >= 434
                control = self._ogm_planner.step(sensor_data, lidar_pgm_image, measurements, control,
                                                 self.COMMANDS_ENUM[directions],
                                                 measurements.player_measurements.transform, target,
                                                 self._enable_steering_rect_using_OGM,
                                                 first_step=(frame==0),
                                                 planner_route_current_cell=self._planner._next_node-1,)

            # comment this, it's used to generate images for episodes routes to create a new experiment suite
            '''if frame == 0:
                return 0, measurement_vec, control_vec, time_out, distance'''

            # Send the control commands to the vehicle
            client.send_control(control)

            # save images if the flag is activated
            self._recording.save_images(sensor_data, episode_name, frame)

            current_x = measurements.player_measurements.transform.location.x
            current_y = measurements.player_measurements.transform.location.y
            current_timestamp = measurements.game_timestamp

            # Get the distance travelled until now
            distance = sldist([current_x, current_y],
                              [target.location.x, target.location.y])

            # Write status of the run on verbose mode
            # Modified: converted to a debug info instead
            info = ''
            # info += '- Route Planner:'
            info += 'Planner: ' + self.COMMANDS_ENUM[directions]
            # info = "Controller is Inputting:"
            info += ' Steer = %f Throttle = %f Brake = %f, Speed = %f' % (control.steer, control.throttle, control.brake,
                                                                          speed)
            # info += '- Status:'
            info += ' [dist=%f] c_x = %f, c_y = %f ---> t_x = %f, t_y = %f' % (float(distance), current_x, current_y,
                                                                               target.location.x, target.location.y)
            logging.debug(info)

            # Check if reach the target
            if distance < self._distance_for_success:
                success = True

            # Increment the vectors and append the measurements and controls.
            frame += 1
            measurement_vec.append(measurements.player_measurements)
            control_vec.append(control)

        if success:
            return 1, measurement_vec, control_vec, float(
                current_timestamp - initial_timestamp) / 1000.0, distance
        return 0, measurement_vec, control_vec, time_out, distance


def run_driving_benchmark(agent,
                          experiment_suite,
                          city_name='Town01',
                          log_name='Test',
                          model_folder_name='-',
                          continue_experiment=False,
                          start_from_exp_pose=(1,1),
                          host='127.0.0.1',
                          port=2000,
                          enable_WZ_avoidance_using_OGM=False,
                          enable_steering_rect_using_OGM=False,
                          simulator_fps=15,
                          visualize_ogm_planner=False,
                          save_ogm_planner_figure=False,
                          start_visualize_or_save_from_frame=0,
                          normal_save_quality=50,
                          save_high_quality_frame_numbers=[],
                          visualize_save_directory=""
                          ):
    while True:
        try:
            with make_carla_client(host, port, timeout=999999) as client:  # 999999999 in Linux, 999999 for Windows
                # Hack to fix for the issue 310, we force a reset, so it does not get
                #  the positions on first server reset.
                client.load_settings(CarlaSettings())
                client.start_episode(0)

                # We instantiate the driving benchmark, that is the engine used to
                # benchmark an agent. The instantiation starts the log process, sets
                
                benchmark = DrivingBenchmark(city_name=city_name,
                                             name_to_save=log_name + '_' + type(experiment_suite).__name__ + '_' +
                                                          model_folder_name + '_' + city_name,
                                             continue_experiment=continue_experiment,
                                             start_from_exp_pose=start_from_exp_pose,
                                             enable_WZ_avoidance_using_OGM=enable_WZ_avoidance_using_OGM,
                                             enable_steering_rect_using_OGM=enable_steering_rect_using_OGM,
                                             simulator_fps=simulator_fps,
                                             visualize_ogm_planner=visualize_ogm_planner,
                                             save_ogm_planner_figure=save_ogm_planner_figure,
                                             start_visualize_or_save_from_frame=start_visualize_or_save_from_frame,
                                             normal_save_quality=normal_save_quality,
                                             save_high_quality_frame_numbers=save_high_quality_frame_numbers,
                                             visualize_save_directory=visualize_save_directory)
                # This function performs the benchmark. It returns a dictionary summarizing
                # the entire execution.

                benchmark_summary = benchmark.benchmark_agent(experiment_suite, agent, client)

                print("")
                print("")
                print("----- Printing results for training weathers (Seen in Training) -----")
                print("")
                print("")
                results_printer.print_summary(benchmark_summary, experiment_suite.train_weathers,
                                              benchmark.get_path())

                print("")
                print("")
                print("----- Printing results for test weathers (Unseen in Training) -----")
                print("")
                print("")

                results_printer.print_summary(benchmark_summary, experiment_suite.test_weathers,
                                              benchmark.get_path())

                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
