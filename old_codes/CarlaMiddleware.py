# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import glob
import os
import sys

try:  # Make sure to install CARLA 0.8.2 (prebuilt version)
    carla_path = os.environ['CARLA_PATH_NEW_VERSION']

    # to import carla (older versions)
    # sys.path.insert(0, carla_path + 'PythonAPI')

    # to import carla (newer versions)
    # sys.path.append(glob.glob(carla_path + 'PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
    #                           (sys.version_info.major, sys.version_info.minor,
    #                            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append(glob.glob(carla_path + 'PythonAPI/carla/dist/carla-*%d.5-%s.egg' % (
        sys.version_info.major, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.insert(0, carla_path + 'PythonAPI/carla')  # to import agents
except IndexError:
    pass

import carla
import logging
import automatic_control
import pygame
import random
import time
# from pynput.keyboard import Key, Controller # For simulating key presses
import subprocess, signal  # For start_CARLA_simlator_ssh
import tensorflow as tf
import queue
import numpy as np
import cv2


# ------------------------------------------------------------------------------
# CARLA Middleware class
# ------------------------------------------------------------------------------
class CarlaMiddleware:
    """
    machine_ip: needed to connect SHH for start_CARLA_simlator_ssh
    game_fps: game_fps=2 for the simulator means each consecutive frames have 1/2 second between them
    """

    def __init__(self, game_fps, host='127.0.0.1', port=2000, gpu_id=0, metadata_filename='metadata.csv',
                 show_carla_clients_gui=True, recording_each_frameNbr=10, change_weather_each_frameNbr=3000,
                 sim_width=640, sim_height=480, client_width=640, client_height=480, reset_if_zeroSpeed_for_time=60,
                 ignore_saving_in_beginning_seconds=30,
                 show_hud=True,
                 graphics_mode='Epic'):
        self.host = host
        self.port = port
        self.gpu_id = gpu_id
        self.metadata_filename = metadata_filename
        self.game_fps = game_fps
        self.ego_vehicle = None
        self.show_carla_clients_gui = show_carla_clients_gui
        self.recording_each_frameNbr = recording_each_frameNbr
        self.change_weather_each_frameNbr = change_weather_each_frameNbr
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.client_width = client_width
        self.client_height = client_height
        self.reset_if_zeroSpeed_for_time = reset_if_zeroSpeed_for_time
        self.ignore_saving_in_beginning_seconds = ignore_saving_in_beginning_seconds
        self.show_hud = show_hud
        self.graphics_mode = graphics_mode
        self.walkers_points = []
        self.vehicles_points = []

    '''
    A function to start CARLA simulator
    init_time: Time to wait after starting CARLA to initialize (in seconds). TODO: make init_time adaptive to wait until simulator is ready, no more than that
    Instead of this you can bash this: cd /home/heraqi/scripts/int-end-to-end-ad/CARLA_0.9.3 ; DISPLAY= ./CarlaUE4.sh /Game/Carla/Maps/Town03 -benchmark -fps=2 -ResX=640 -ResY=480 -quality-level=Epic --avoid-stopping
    '''

    def start_CARLA_simlator_ssh(self, town='01', show_carla_server_gui=False, init_time=10.0):
        # Kill the server  (simulator) if already running
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'CarlaUE4' in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)

        # Run Carla server (simulator)
        bashCommand = carla_path + 'CarlaUE4.sh Town' + ' -windowed -ResX=' + str(self.sim_width) + \
                      ' -ResY=' + str(self.sim_height) + ' -carla-rpc-port=' + str(self.port) + \
                      ' -timeout=1000000ms' + ' -fixed_delta_seconds=' + str(1. / self.game_fps) + \
                      '-quality-level=' + self.graphics_mode  # -carla-server
        # bashCommand = 'vglrun -d :7.' + str(gpu_id) + ' ' + bashCommand
        # If headless start of the simulator (no real window; headless)
        if not show_carla_server_gui:
            # setting the environment variable DISPLAY to empty
            bashCommand = 'DISPLAY= ' + bashCommand
        print("CARLA server command called: " + bashCommand)
        # Command: DISPLAY= /home/heraqi/CARLA_0.8.2/CarlaUE4.sh /Game/Maps/Town01 -windowed -ResX=1280
        #   -ResY=720 -carla-port=2000 -timeout=1000000ms -carla-server -benchmark -fps=15-quality-level=Epic
        FNULL = open(os.devnull, 'w')
        print("CARLA server launch command called: " + bashCommand)
        process = subprocess.Popen(bashCommand, shell=True, stdout=FNULL, stderr=subprocess.STDOUT,
                                   preexec_fn=os.setpgrp)
        # output, error = process.communicate()
        # print('STDOUT:{}'.format(output))
        time.sleep(init_time)  # Wait until CARLA is initialized

        # Fill client data
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10000.0)  # should be big enough because PC maybe loaded
        self.world = self.client.load_world('Town' + town)  # self.client.get_world()

        self.map = self.world.get_map()
        self.blueprints = self.world.get_blueprint_library().filter(
            'vehicle.*')  # Full actor list even from possible previous Python runs
        self.created_actor_list = []  # current Python run created actor list

        # World settings
        self.settings = self.world.get_settings()
        # self.settings.synchronous_mode = True  # Set synchronous mode
        # self.world.apply_settings(self.settings)

    def get_Carla_mw_world(self, mode='Roaming', with_sensors=True, recording=False, out_folder_episode='',
                           record_data_for_time=3600):
        pygame.init()
        pygame.font.init()
        hud = automatic_control.HUD(self.client_width, self.client_height, self.show_hud)
        world = automatic_control.World(self.world, hud, self.game_fps, ego_vehicle=self.ego_vehicle,
                                        recording=recording,
                                        record_data_for_time=record_data_for_time,
                                        pov_camera=True,
                                        out_folder_episode=out_folder_episode,
                                        metadata_filename=self.metadata_filename,
                                        recording_each_frameNbr=self.recording_each_frameNbr,
                                        change_weather_each_frameNbr=self.change_weather_each_frameNbr,
                                        reset_if_zeroSpeed_for_time=self.reset_if_zeroSpeed_for_time,
                                        ignore_saving_in_beginning_seconds=self.ignore_saving_in_beginning_seconds,
                                        show_carla_clients_gui=self.show_carla_clients_gui)
        return world

    def spawn_pedestrians(self, nPedestrians=30):
        pedestrians_points = []
        for i in range(nPedestrians):
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            if (spawn_point.location != None):
                pedestrians_points.append(spawn_point)
        print('Found %d pedestrian spawn points' % len(pedestrians_points))

        pedestrian_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        # Build the batch of commands to spawn the pedestrians
        batch = []
        for spawn_point in pedestrians_points:
            walker_bp = random.choice(pedestrian_blueprints)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        # apply the batch
        results = self.client.apply_batch_sync(batch, True)
        pedestrians_list = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                pedestrians_list.append({"id": results[i].actor_id})

        # Spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(pedestrians_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), pedestrians_list[i]["id"]))
        # apply the batch
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                pedestrians_list[i]["con"] = results[i].actor_id

        # Put altogether the walkers and controllers id to get the objects from their id
        all_id = []
        for i in range(len(pedestrians_list)):
            all_id.append(pedestrians_list[i]["con"])
            all_id.append(pedestrians_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        # Initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_actors), 2):
            all_actors[i].start()  # start walker
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())  # set walk to random point
            # random max speed
            all_actors[i].set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
        print('spawned %d pedestrians, press Ctrl+C to exit.' % nPedestrians)

    '''
    avoids spawning vehicles prone to accidents
    '''
    def spawn_vehicles(self, nVehicles=20, delay=2.0):
        self.vehicles_points = list(self.map.get_spawn_points())
        random.shuffle(self.vehicles_points)
        print('Found %d vehicle spawn points' % len(self.vehicles_points))

        vehicle_blueprints = [x for x in self.blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith('isetta')]

        def try_spawn_random_vehicle_at(transform):
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                self.created_actor_list.append(vehicle)
                vehicle.set_autopilot()
                print('spawned vehicle %r at (%s %s %s)' % (
                    vehicle.type_id, transform.location.x, transform.location.y, transform.location.z))
                return True
            return False

        # Wait until 'try_spawn_random_vehicle_at' trials succeeds in completing the nVehicles spawns
        count = nVehicles
        for spawn_point in self.vehicles_points:
            # time.sleep(delay)  # TODO: check if not neeeded remove it
            if try_spawn_random_vehicle_at(spawn_point):
                count -= 1
            if count <= 0:
                break
        print('spawned %d vehicles, press Ctrl+C to exit.' % nVehicles)

        # Suggest ego car location such that it doesn't collide with other cars
        print('trying to spawn ego vehicle.')
        while True:
            time.sleep(delay)
            self.ego_vehicle = try_spawn_random_vehicle_at(random.choice(self.vehicles_points))
            if self.ego_vehicle:
                print('ego vehicle spawned')
                return

    '''
    TODO: destroy all other actors inlcuding like sensors, not only vehicles
    '''

    def reset_simulator(self):
        # Filtering is important as full list has lots of other actors
        vehicles = self.world.get_actors().filter('vehicle.*')
        walkers = self.world.get_actors().filter('walker.*')
        print('\ndestroying %d actors' % len(vehicles))
        for vehicle in vehicles:
            vehicle.destroy()
        for walker in walkers:
            walker.destroy()

    def create_move_ego_vehicle_w_sensors(self, mode='autopilot', with_sensors=True, recording=False,
                                          out_folder_episode='', record_data_for_time=3600):
        pygame.init()
        display = None
        if self.show_carla_clients_gui:
            display = pygame.display.set_mode((self.client_width, self.client_height), pygame.HWSURFACE |
                                              pygame.DOUBLEBUF)

        hud = automatic_control.HUD(self.client_width, self.client_height, self.show_hud)
        world = automatic_control.World(self.world, hud, self.game_fps, ego_vehicle=self.ego_vehicle,
                                        recording=recording,
                                        record_data_for_time=record_data_for_time,
                                        pov_camera=True,
                                        out_folder_episode=out_folder_episode,
                                        metadata_filename=self.metadata_filename,
                                        recording_each_frameNbr=self.recording_each_frameNbr,
                                        change_weather_each_frameNbr=self.change_weather_each_frameNbr,
                                        reset_if_zeroSpeed_for_time=self.reset_if_zeroSpeed_for_time,
                                        ignore_saving_in_beginning_seconds=self.ignore_saving_in_beginning_seconds,
                                        show_carla_clients_gui=self.show_carla_clients_gui)

        # Uncomment to enable if keyboard is needed
        agent = None
        if mode == "autopilot":
            keyboard_controller = automatic_control.KeyboardControl(world, start_in_autopilot=True)
        else:
            keyboard_controller = automatic_control.KeyboardControl(world, start_in_autopilot=False)
            if mode == "roaming":
                agent = automatic_control.RoamingAgent(world.vehicle)
            elif mode == "destination_pid":
                agent = automatic_control.BasicAgent(world.vehicle)
                # get random point # TODO make use of destination parameter
                spawn_point = world.world.get_map().get_spawn_points()[0]
                agent.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))
            else:
                print("Unknown value for ego car navigation mode.")
                return

        # Data collection infinite loop (Game Loop)
        frame = None
        clock = pygame.time.Clock()
        world.next_weather()  # To select the first training weather at the beginning
        while True:
            clock.tick()  # Needed ?
            if self.settings.synchronous_mode:
                world.world.tick()  # Needed if carla.Client(...).get_world.get_settings().synchronous_mode = True
            # If server is ready (provided a new frame), takes the timeout as parameter
            if world.world.wait_for_tick(10.0):
                world.update_hud(clock)  # Advance world class in automatic_control, the HUD is updated

                # Capture & show world (world simulator is already running)
                if (self.show_carla_clients_gui):
                    # Uncomment to enable if keyboard is needed
                    if (keyboard_controller is not None) and keyboard_controller.parse_events(world, clock):
                        return

                    world.render(display)  # capture updated world image from the server (simulator)
                    pygame.display.flip()  # show the world

                # Apply control
                if agent is not None:
                    control = agent.run_step()
                    control.manual_gear_shift = False
                    world.vehicle.apply_control(control)

        if world is not None:
            world.destroy()
        pygame.quit()

    def deploy_model_to_ego_vehicle_w_sensors(self, networkTensors, tf_session, experiment_configs, deploy_for_time,
                                              action_each_frameNbr, ignore_actions_in_beginning_seconds,
                                              il_model=False):
        # TODO: deploy_for_time is not used
        pygame.init()
        pygame.font.init()
        if (self.show_carla_clients_gui):
            display = pygame.display.set_mode((self.client_width, self.client_height), pygame.HWSURFACE |
                                              pygame.DOUBLEBUF)

        # Create data from Carla queues
        carla_queues = {"cam1_data_queue": queue.Queue(), "cam2_data_queue": queue.Queue(),
                        "cam3_data_queue": queue.Queue(), "lidar_data_queue": queue.Queue(),
                        "metadata_queue": queue.Queue()}

        hud = automatic_control.HUD(self.client_width, self.client_height, self.show_hud)
        world = automatic_control.World(self.world, networkTensors, hud, self.game_fps,
                                        action_each_frameNbr=action_each_frameNbr,
                                        ego_vehicle=self.ego_vehicle,
                                        deploy_for_time=deploy_for_time,
                                        change_weather_each_frameNbr=self.change_weather_each_frameNbr,
                                        reset_if_zeroSpeed_for_time=self.reset_if_zeroSpeed_for_time,
                                        ignore_actions_in_beginning_seconds=
                                        ignore_actions_in_beginning_seconds,
                                        show_carla_clients_gui=self.show_carla_clients_gui,
                                        carla_queues=carla_queues)

        # Uncomment to enable if keyboard is needed
        controller = automatic_control.KeyboardControl(world, start_in_autopilot=False)

        # Deployment infinite loop (Game Loop)
        frame = None
        clock = pygame.time.Clock()
        world.next_weather()  # To select the first training weather at the beginning # TODO: should be testing weather
        while True:
            clock.tick()  # Needed ?
            if self.settings.synchronous_mode:
                world.world.tick()  # Needed if carla.Client(...).get_world.get_settings().synchronous_mode = True
            else:
                print("carla.Client(...).get_world.get_settings().synchronous_mode should True!")
                return

            ts = world.world.wait_for_tick()  # server is ready (provided a new frame)

            # Capture & show world (world simulator is already running)
            if (self.show_carla_clients_gui):
                # Uncomment to enable if keyboard is needed
                # if controller.parse_events(world, clock):
                #   return
                world.render(display)  # capture updated world image from the server (simulator)
                pygame.display.flip()  # show the world

            # Make sure data is ready
            # if all([q.empty() for q in carla_queues.values()]):
            if carla_queues["cam1_data_queue"].empty() or carla_queues["metadata_queue"].empty():
                print("Simulator ready but didn't provide data or maybe waiting for \"ignore_actions_in_beginning\"!")
                continue

            # Read data
            image_1_carla = carla_queues["cam1_data_queue"].get(timeout=2.0)
            metadata = carla_queues["metadata_queue"].get()
            speed = int(metadata["speed"])
            if image_1_carla.frame_number != int(metadata["data_file_name"]):
                print("Inconsistent data from different sensors received from the simulator!")
                continue

            # Read and resize image
            array = np.frombuffer(image_1_carla.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image_1_carla.height, image_1_carla.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            image_1 = cv2.resize(array, dsize=(experiment_configs.img_width, experiment_configs.img_height),
                                 interpolation=cv2.INTER_CUBIC)

            # Prepare feedDict
            if il_model:
                cmd = 2
                inputData = []
                inputData.append(tf_session.run(tf.one_hot(cmd, 4)))  # 2 for follow lane
                inputData.append([[speed]])  # Speed
                feedDict = {networkTensors['inputs'][0]: [image_1],
                            networkTensors['inputs'][1][0]: [inputData[0]],
                            networkTensors['inputs'][1][1]: inputData[1],
                            networkTensors['dropoutVec']: experiment_configs.dropout_vec
                            }
            else:
                cmd = 0
                feedDict = {networkTensors['inputs'][0]: [image_1],
                            networkTensors['inputs'][1]: [[speed]]}

            # Run neural network
            if il_model:
                targets = tf_session.run(networkTensors['output']['output'], feed_dict=feedDict)
                steer, throttle, brake = targets[cmd][0]  # First indexing [0] means follow-lane branch is assumed
                # speed = targets[1]  # TODO: speed branch was commented during training branchConfigs
            else:
                targets = tf_session.run(networkTensors['network_branches'], feed_dict=feedDict)
                steer, throttle, brake = targets[cmd][0]  # First indexing [0] means follow-lane branch is assumed
                # steer, throttle, brake = (0, 1, 0)  # For debugging
                # speed = targets[1]  # TODO: speed branch was commented during training branchConfigs

            # Apply control resulted from network
            # TODO: brake is ignore for now
            # world.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer),
            #                                                  brake=float(brake)))
            world.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), steer=float(steer)))

            # Advance world class in automatic_control, the HUD is updated
            print("Vehicle moved using provided network data")
            world.update_hud(clock)

        # world.destroy()
        # pygame.quit()
