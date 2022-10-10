#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import numpy as np
import csv  # to write metainfo data file
import queue

# find carla module
import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent


# ==============================================================================
# -- Global functions
# ==============================================================================
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World for data collection
# ==============================================================================
# Handles weather & camera sensor with CameraManager
class World(object):
    def __init__(self, carla_world, hud, game_fps, ego_vehicle=None, start_point=None, recording=True,
                 record_data_for_time=3600, pov_camera=True,
                 out_folder_episode='_out', metadata_filename='metadata.csv',
                 recording_each_frameNbr=10, change_weather_each_frameNbr=3000, reset_if_zeroSpeed_for_time=60,
                 ignore_saving_in_beginning_seconds=30,
                 show_carla_clients_gui=True, deploy=False, agent_model=None):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.game_fps = game_fps  # 1 / game time (real-time of world) between each two server consecutive frames
        self.vehicle = None
        self.collision_sensor = None
        self.gnss_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.change_weather_each_frameNbr = change_weather_each_frameNbr
        self.reset_if_zeroSpeed_for_frames = reset_if_zeroSpeed_for_time * game_fps
        self.ignore_saving_in_beginning_frames = ignore_saving_in_beginning_seconds * game_fps
        self.recording = recording
        self.record_data_for_frames = record_data_for_time * game_fps
        self.pov_camera = pov_camera
        self.start_point = start_point
        self.ego_vehicle = ego_vehicle
        self.out_folder_episode = out_folder_episode
        self.deploy = deploy
        self.agent_model = agent_model

        self.csv_file = open(out_folder_episode + '/' + metadata_filename, 'w+')
        self.csv_writer = csv.writer(self.csv_file, delimiter=',')
        self.csv_writer.writerow(
            ['data_file_name', 'speed_kmh', 'heading', 'location', 'GNSS', 'throttle', 'steer', 'brake', 'reverse',
             'hand_brake', 'manual', 'gear', 'collision', 'nbr_vehicles', 'weather'])

        self.show_carla_clients_gui = show_carla_clients_gui
        self.recording_each_frameNbr = recording_each_frameNbr
        self.restart()
        self.world.on_tick(self.on_world_tick)  # Call back from the simulator each tick (each new frame is provided)

    # Call back from the simulator each tick (each new frame is provided)
    def on_world_tick(self, timestamp):
        # Data collection finished
        if timestamp.frame_count >= self.record_data_for_frames:
            print("Data collection finished, exiting running script!")
            sys.exit(0)  # TODO: better closing
        # Change weather
        if timestamp.frame_count % self.change_weather_each_frameNbr == 0:
            self.next_weather()
        self.hud.on_world_tick(timestamp)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None \
            else (1 if self.pov_camera == True else 0)

        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the vehicle.
        if self.vehicle == None:
            if self.vehicle is not None:
                spawn_point = self.vehicle.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                self.destroy()

            while self.vehicle is None:
                if (self.start_point is not None):
                    spawn_point = carla.Transform(self.start_point)
                else:
                    spawn_points = self.world.get_map().get_spawn_points()
                    random.shuffle(spawn_points)
                    try:
                        self.vehicle = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
                    except:
                        continue
        else:
            self.vehicle = self.ego_vehicle

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.gnss_sensor = GnssSensor(self.vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self, self.hud, self.recording,
                                            out_folder_episode=self.out_folder_episode,
                                            recording_each_frameNbr=self.recording_each_frameNbr,
                                            ignore_saving_in_beginning_frames=self.ignore_saving_in_beginning_frames,
                                            show_carla_clients_gui=self.show_carla_clients_gui)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.create_sensors(deployment=False)
        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)

    '''
    Data is collected according to CARLA benchmark (Alexey Dosovitskiy, "CARLA: An Open Urban Driving Simulator") only 
    training weather conditions ([1:ClearNoon, 3:WetNoon, 6:HardRainNoon, 8:ClearSunset], test were [4:WetCloudyNoon, 14:SoftRainSunset])
    0 - Default
    1 - ClearNoon
    2 - CloudyNoon
    3 - WetNoon
    4 - WetCloudyNoon
    5 - MidRainyNoon
    6 - HardRainNoon
    7 - SoftRainNoon
    8 - ClearSunset
    9 - CloudySunset
    10 - WetSunset
    11 - WetCloudySunset
    12 - MidRainSunset
    13 - HardRainSunset
    14 - SoftRainSunset
    '''

    def next_weather(self, reverse=False):
        weathers = [self._weather_presets[i] for i in [0, 12, 4, 1]]  # Training
        # weathers = [self._weather_presets[i] for i in [10, 9]] # Test or Validation
        weather = random.choice(weathers)
        self.hud.notification('Weather: %s' % weather[1])
        self.vehicle.get_world().set_weather(weather[0])

    def update_hud(self, clock):
        self.hud.update_hud(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

        # Reset simulator if car stopped for long time (CARLA deadlock reached)
        # Recent speeds at the beginning (FIFO buffer)
        valid_speeds = list(filter(lambda v: v != None, self.camera_manager.previous_speeds))
        if len(valid_speeds) > self.reset_if_zeroSpeed_for_frames:
            recent_speeds = valid_speeds[0: self.reset_if_zeroSpeed_for_frames]
            # recent_speeds = [round(x) for x in recent_speeds] # Round speeds if needed
            if all(v <= 0 for v in recent_speeds):  # Car stopped for reset_if_zeroSpeed_for_time seconds of game time
                # Reset the data collection
                print("Car stopped for long time, restarting data collection!")
                # TODO: soft restart instead of restarting the script
                # TODO: causes the script to exit if called with nohop python ..... &
                os.execv(sys.executable, ['python'] + sys.argv + ['--nodelete'])

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.gnss_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- World for deployment
# ==============================================================================
# Handles weather & camera sensor with CameraManager
# TODO: deployment=True is not handled properly yet
class World_Deployment(object):
    def __init__(self, carla_world, networkTensors, hud, game_fps, action_each_frameNbr=10, ego_vehicle=None,
                 start_point=None, deploy_for_time=3600, change_weather_each_frameNbr=3000,
                 reset_if_zeroSpeed_for_time=60, ignore_actions_in_beginning_seconds=30, show_carla_clients_gui=True,
                 pov_camera=True, carla_queues=None, deployment=False):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.game_fps = game_fps  # 1 / game time (real-time of world) between each two server consecutive frames
        self.vehicle = None
        self.collision_sensor = None
        self.gnss_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.change_weather_each_frameNbr = change_weather_each_frameNbr
        self.ignore_actions_in_beginning_frames = ignore_actions_in_beginning_seconds * game_fps
        self.reset_if_zeroSpeed_for_frames = reset_if_zeroSpeed_for_time * game_fps
        self.deploy_for_frames = deploy_for_time * game_fps
        self.pov_camera = pov_camera
        self.start_point = start_point
        self.ego_vehicle = ego_vehicle
        self.networkTensors = networkTensors
        self.show_carla_clients_gui = show_carla_clients_gui
        self.action_each_frameNbr = action_each_frameNbr
        self.carla_queues = carla_queues
        self.deployment = deployment

        self.restart()
        self.world.on_tick(self.on_world_tick)  # Call back from the simulator each tick (each new frame is provided)

    # Call back from the simulator each tick (each new frame is provided)
    def on_world_tick(self, timestamp):
        # Data collection finished
        if timestamp.frame_count >= self.deploy_for_frames:
            print("Deployment finished, exiting running script!")
            sys.exit(0)  # TODO: better closing
        # Change weather
        if timestamp.frame_count % self.change_weather_each_frameNbr == 0:
            self.next_weather()
        self.hud.on_world_tick(timestamp)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else (
            1 if self.pov_camera == True else 0)

        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the vehicle.
        if self.vehicle == None:
            if self.vehicle is not None:
                spawn_point = self.vehicle.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
                self.destroy()

            while self.vehicle is None:
                if (self.start_point is not None):
                    spawn_point = carla.Transform(self.start_point)
                else:
                    spawn_points = self.world.get_map().get_spawn_points()
                    random.shuffle(spawn_points)
                    try:
                        self.vehicle = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
                    except:
                        continue
        else:
            self.vehicle = self.ego_vehicle

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.gnss_sensor = GnssSensor(self.vehicle)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self, self.hud,
                                            show_carla_clients_gui=self.show_carla_clients_gui,
                                            action_each_frameNbr=self.action_each_frameNbr,
                                            ignore_actions_in_beginning_frames=self.ignore_actions_in_beginning_frames,
                                            carla_queues=self.carla_queues)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.create_sensors(deployment=self.deployment)
        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)

    '''
    Weathers:
    0 - Default
    1 - ClearNoon
    2 - CloudyNoon
    3 - WetNoon
    4 - WetCloudyNoon
    5 - MidRainyNoon
    6 - HardRainNoon
    7 - SoftRainNoon
    8 - ClearSunset
    9 - CloudySunset
    10 - WetSunset
    11 - WetCloudySunset
    12 - MidRainSunset
    13 - HardRainSunset
    14 - SoftRainSunset
    '''

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def update_hud(self, clock):
        self.hud.update_hud(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

        # Reset simulator if car stopped for long time (CARLA deadlock reached)
        # Recent speeds at the beginning (FIFO buffer)
        valid_speeds = list(filter(lambda v: v != None, self.camera_manager.previous_speeds))
        if len(valid_speeds) > self.reset_if_zeroSpeed_for_frames:
            recent_speeds = valid_speeds[0: self.reset_if_zeroSpeed_for_frames]
            # recent_speeds = [round(x) for x in recent_speeds] # Round speeds if needed
            if all(v <= 0 for v in recent_speeds):  # Car stopped for reset_if_zeroSpeed_for_time seconds of game time
                # Reset the data collection
                print("Car stopped for long time, restarting data collection!")
                # TODO: soft restart instead of restarting the script
                # TODO: causes the script to exit if called with nohop python ..... &
                os.execv(sys.executable, ['python'] + sys.argv + ['--nodelete'])

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.gnss_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl
# ==============================================================================
class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        world.vehicle.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.show_next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.show_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = world.vehicle.get_control().gear
                    world.hud.notification('%s Transmission' % (
                        'Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.vehicle.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            self._parse_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD
# ==============================================================================
class HUD(object):
    def __init__(self, width, height, show_hud):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height, show_hud)
        self.server_fps = 0  # 1 / machine actual display time between each two server consecutive frames
        self.frame_number = 0
        self.simulation_time = 0
        self.show_hud = show_hud
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()  # Actual FPS, based on machine specs
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def update_hud(self, world, clock):
        if not self._show_info:
            return
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time (Game time): % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, -t.location.y)),  # y should be inverted
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            def distance(l):
                return math.sqrt(
                    (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)

            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        if (self._show_info):
            self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self.show_hud:
            if self._show_info:
                info_surface = pygame.Surface((220, self.dim[1]))
                info_surface.set_alpha(100)
                display.blit(info_surface, (0, 0))
                v_offset = 4
                bar_h_offset = 100
                bar_width = 106
                for item in self._info_text:
                    if v_offset + 18 > self.dim[1]:
                        break
                    if isinstance(item, list):
                        if len(item) > 1:
                            points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                            pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                        item = None
                        v_offset += 18
                    elif isinstance(item, tuple):
                        if isinstance(item[1], bool):
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                            pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                        else:
                            rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                            pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                            f = (item[1] - item[2]) / (item[3] - item[2])
                            if item[2] < 0.0:
                                rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                            else:
                                rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                            pygame.draw.rect(display, (255, 255, 255), rect)
                        item = item[0]
                    if item:  # At this point has to be a str.
                        surface = self._font_mono.render(item, True, (255, 255, 255))
                        display.blit(surface, (8, v_offset))
                    v_offset += 18
            self._notifications.render(display)
            self.help.render(display)


# ==============================================================================
# -- FadingText
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    def __init__(self, font, width, height, show_help):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        self.show_help = show_help
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render and self.show_help:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, world, hud, recording=False, out_folder_episode="", recording_each_frameNbr=None,
                 action_each_frameNbr=None, ignore_saving_in_beginning_frames=None,
                 ignore_actions_in_beginning_frames=None, show_carla_clients_gui=False, carla_queues=None):
        self.sensor = None
        self._surface = None
        self.sensors = []
        self._parent = parent_actor
        self.world = world
        self._hud = hud
        self._recording = recording
        self.out_folder_episode = out_folder_episode
        self.world = world  # Needed to access world.vehicle sensors (like collisions) and store it as metadata
        self.recording_each_frameNbr = recording_each_frameNbr
        self.action_each_frameNbr = action_each_frameNbr
        self.ignore_saving_in_beginning_frames = ignore_saving_in_beginning_frames
        self.ignore_actions_in_beginning_frames = ignore_actions_in_beginning_frames
        self.show_carla_clients_gui = show_carla_clients_gui
        self.carla_queues = carla_queues

        # Recent speeds at the beginning (FIFO buffer)
        self.previous_speeds = [
                                   None] * 10000  # TODO: length should be propotional to set World.reset_if_zeroSpeed_for_time
        # Sensors positions relative to the vehicle TODO: Each sensor should be associated with unique position relative to car
        self._camera_transforms = [
            # carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.5, y=0, z=1.5), carla.Rotation(yaw=0)),
            carla.Transform(carla.Location(x=1.5, y=1.99, z=1.5), carla.Rotation(yaw=30)),
            carla.Transform(carla.Location(x=1.5, y=-1.99, z=1.5), carla.Rotation(yaw=-30)),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=1.6, z=2.5)),
        ]
        self._transform_index = 0  # default client shown camera
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Front'],  # Front
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Right'],  # Right
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Left'],  # Left
            # ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            # ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            if item[0].startswith('sensor.lidar'):
                bp.set_attribute('channels', '32')  # 32 is the default
                bp.set_attribute('range', '15000')  # 1000 is the default (means 10 meter?)
                bp.set_attribute('upper_fov', '10')  # 10 is the default
                bp.set_attribute('lower_fov', '-30')  # -30 is the default
                # 10 is the default (how many 360 full rotations happen per second). A full rotation is composed of
                # some frames based on FPS (game)
                bp.set_attribute('rotation_frequency', '15')
                # 56000 is the default, larger values means smaller Horizontal angle per frame
                # For 1 horizontal angle step, points per full rotation = 32 channel * 360/1
                # Points per second = 15*32*360/1 = 172800
                # For 1 horizontal angle: 172800/1 = 172800
                bp.set_attribute('points_per_second', '172800')

            item.append(bp)
        self._shown_sensor_index = 0

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        # TODO: enable Tab press to change displayed view (without affecting saved data)
        # self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def create_sensors(self, deployment=False):
        if deployment:
            parse_image_function = CameraManager._parse_image_deployment
        else:
            parse_image_function = CameraManager._parse_image

        nbrCreatedSensors = 0
        for index in range(len(self._sensors)):
            self.sensors.append(self._parent.get_world().spawn_actor(self._sensors[index][-1],
                                                                     self._camera_transforms[index],
                                                                     attach_to=self._parent))
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            # (circular reference)
            weak_self = weakref.ref(self)
            # listen provides one variable with data to a callback: lamda anyname: <code that uses anyname>
            # TODO: call the function once with sensor_index as variable instead of handcoding them. Take care that
            #  these handcoded indices should align with the "_sensor" list order of sensors deficntion.

            if self._sensors[index][0].startswith("sensor.camera.depth"):
                self.sensors[index].listen(
                    lambda object_from_sensor: parse_image_function(weak_self, 3, object_from_sensor,
                                                                    self.out_folder_episode,
                                                                    self.show_carla_clients_gui, nbrCreatedSensors))
            elif self._sensors[index][0].startswith("sensor.camera.rgb"):
                if self._sensors[index][2] == "Camera RGB Front":
                    self.sensors[index].listen(
                        lambda object_from_sensor: parse_image_function(weak_self, 0, object_from_sensor,
                                                                        self.out_folder_episode,
                                                                        self.show_carla_clients_gui,
                                                                        nbrCreatedSensors))
                elif self._sensors[index][2] == "Camera RGB Right":
                    self.sensors[index].listen(
                        lambda object_from_sensor: parse_image_function(weak_self, 1, object_from_sensor,
                                                                        self.out_folder_episode,
                                                                        self.show_carla_clients_gui,
                                                                        nbrCreatedSensors))
                elif self._sensors[index][2] == "Camera RGB Left":
                    self.sensors[index].listen(
                        lambda object_from_sensor: parse_image_function(weak_self, 2, object_from_sensor,
                                                                        self.out_folder_episode,
                                                                        self.show_carla_clients_gui,
                                                                        nbrCreatedSensors))
            elif self._sensors[index][0].startswith("sensor.camera.semantic_segmentation"):
                self.sensors[index].listen(
                    lambda object_from_sensor: parse_image_function(weak_self, 4, object_from_sensor,
                                                                    self.out_folder_episode,
                                                                    self.show_carla_clients_gui,
                                                                    nbrCreatedSensors))
            elif self._sensors[index][0].startswith("sensor.lidar"):
                self.sensors[index].listen(
                    lambda object_from_sensor: parse_image_function(weak_self, 5, object_from_sensor,
                                                                    self.out_folder_episode,
                                                                    self.show_carla_clients_gui,
                                                                    nbrCreatedSensors))
            nbrCreatedSensors = nbrCreatedSensors + 1

        self.sensor = self.sensors[0]

    def show_sensor(self, index):
        index = index % len(self._sensors)
        self._hud.notification(self._sensors[index][2])
        self._shown_sensor_index = index

    def show_next_sensor(self):
        self.show_sensor(self._shown_sensor_index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def store_metadata(weak_self, frame_number):
        self = weak_self()

        world = self.world
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()

        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        # Add to previous speeds (needed to detect if ego car stops for long time)
        spd = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self.previous_speeds.pop()
        self.previous_speeds.insert(0, round(spd))  # Comment round if precision is needed

        data_file_name = '%08d.png' % frame_number
        speed = '%3.3f' % spd
        heading = '%3.3f\N{DEGREE SIGN} %2s' % (t.rotation.yaw, heading)
        location = '(%5.1f, %5.1f)' % (t.location.x, -t.location.y)  # y is inverted
        gnss = '(%2.6f, %3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)
        gear = '%s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)
        throttle = '%3.3f' % c.throttle
        steer = '%3.3f' % c.steer
        brake = '%3.3f' % c.brake
        reverse = '%3.3f' % c.reverse
        hand_brake = '%3.3f' % c.hand_brake
        manual_gear_shift = '%3.3f' % c.manual_gear_shift
        weather = world._weather_presets[world._weather_index][1]  # [1] at the tuple means the string

        # data_file_name, speed_kmh, heading, location, gnss, throttle, steer, brake, reverse, hand_brake, manual,
        # gear, collision, nbr_vehicles, weather
        world.csv_writer.writerow([data_file_name, speed, heading, location, gnss, throttle, steer, brake, reverse,
                                   hand_brake, manual_gear_shift, gear, 0, len(vehicles), weather])
        world.csv_file.flush()

    # because save_to_disk for lidarmeasurment class doesnt' work in CARLA 0.9.6
    @staticmethod
    def save_lidar_to_disk(data, filename):
        """Save this point-cloud to disk as PLY format."""
        filename = filename + '.ply'
        def construct_ply_header():
            """Generates a PLY header given a total number of 3D points and
            coloring property if specified
            """
            points = len(data)  # Total point number
            header = ['ply',
                      'format ascii 1.0',
                      'element vertex {}',
                      'property float32 x',
                      'property float32 y',
                      'property float32 z',
                      'property uchar diffuse_red',
                      'property uchar diffuse_green',
                      'property uchar diffuse_blue',
                      'end_header']
            return '\n'.join(header[0:6] + [header[-1]]).format(points)

        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        ply = '\n'.join(['{:.2f} {:.2f} {:.2f}'.format(*p) for p in points.tolist()])

        # Create folder to save if does not exist.
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Open the file and save with the specific PLY format.
        with open(filename, 'w+') as ply_file: ply_file.write('\n'.join([construct_ply_header(), ply]))

    # Callback for training data collection
    @staticmethod
    # TODO: because it's a callback from CARLA lib and static, no breakpoints possible and doesn't cause the main code
    #  to break in case of errors
    def _parse_image(weak_self, sensor_index, object_from_sensor, out_folder_episode, display_client_gui,
                     nbrCreatedSensors):
        self = weak_self()
        if not self:
            return
        if nbrCreatedSensors < len(self._sensors):
            return

        # Prepare data
        data_type = self._sensors[sensor_index][1]
        if data_type is not None:
            object_from_sensor.convert(data_type)

        # Save (and then render if display_client_gui) each recording_each_frameNbr
        if object_from_sensor.frame % self.recording_each_frameNbr == 0:
            # Save data
            if self._recording and object_from_sensor.frame > self.ignore_saving_in_beginning_frames:
                # Save data, for LiDAR sensor object_from_sensor, it saves ply file

                # because save_to_disk for lidarmeasurment class doesnt' work in CARLA 0.9.6
                # TODO: when they fix that call object_from_sensor.save_to_disk always without if condition and
                #  remvoe the save_lidar_to_disk function
                if self._sensors[sensor_index][0].startswith('sensor.lidar'):
                    CameraManager.save_lidar_to_disk(object_from_sensor,
                        out_folder_episode + '/%s/%08d' % (self._sensors[sensor_index][0], object_from_sensor.frame))
                else:
                    object_from_sensor.save_to_disk(
                        out_folder_episode + '/%s/%08d' % (self._sensors[sensor_index][0], object_from_sensor.frame))

                # If front RGB Camera store meta information
                if sensor_index == 0:
                    self.store_metadata(weak_self, object_from_sensor.frame)

        # Display client
        if display_client_gui and self._shown_sensor_index == sensor_index:
            if self._sensors[sensor_index][0].startswith('sensor.lidar'):
                points = np.frombuffer(object_from_sensor.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                # x,y,z: +x is to the left of the car, +y is to the back of the car, +z is to the ground downwards.
                lidar_data = np.array(points[:, :2])
                lidar_data *= min(self._hud.dim) / 400.0  # TODO: denominator should be high for larger LiDAR ranges
                lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
                lidar_data = np.fabs(lidar_data)
                lidar_data = lidar_data.astype(np.int32)
                lidar_data = np.reshape(lidar_data, (-1, 2))
                # print number of scan points for debugging, it differs because of non reflections due to limited
                # range
                # print(len(lidar_data))
                lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
                lidar_img = np.zeros(lidar_img_size)
                lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
                self._surface = pygame.surfarray.make_surface(lidar_img)
            else:
                array = np.frombuffer(object_from_sensor.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (object_from_sensor.height, object_from_sensor.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    @staticmethod
    def get_metadata(weak_self, frame_number):
        self = weak_self()

        world = self.world
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()

        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        # Add to previous speeds (needed to detect if ego car stops for long time)
        spd = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self.previous_speeds.pop()
        self.previous_speeds.insert(0, round(spd))  # Comment round if precision is needed

        data = {"data_file_name": frame_number,
                "speed": spd,
                "heading": (t.rotation.yaw, heading),
                "location_x": t.location.x,
                "location_y": -t.location.y,  # y is inverted
                "gnss": (world.gnss_sensor.lat, world.gnss_sensor.lon),
                "gear": {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
                "throttle": c.throttle,
                "steer": c.steer,
                "brake": c.brake,
                "reverse": c.reverse,
                "hand_brake": c.hand_brake,
                "manual_gear_shift": c.manual_gear_shift,
                "weather": world._weather_presets[world._weather_index][1]}  # [1] at the tuple means the string

        return data

    # Callback for deployment data collection
    @staticmethod
    # TODO: because it's a callback from CARLA lib and static, no breakpoints possible and doesn't cause the main code
    #  to break in case of errors
    # TODO: needs updates to bug fixes as done in _parse_image function
    def _parse_image_deployment(weak_self, sensor_name, object_from_sensor, out_folder_episode, display_client_gui,
                                nbrCreatedSensors):
        self = weak_self()
        if not self:
            return
        if nbrCreatedSensors < len(self._sensors):
            return
        # Save (and then render if display_client_gui) each recording_each_frameNbr
        if object_from_sensor.frame % self.action_each_frameNbr == 0:
            # Prepare data
            sensor_index = [self._sensors[i][0].startswith(sensor_name.rstrip('0123456789')) for i in
                            range(len(self._sensors))].index(True)
            data_type = self._sensors[sensor_index][1]
            if data_type is not None:
                object_from_sensor.convert(data_type)

            # Save data
            if object_from_sensor.frame > self.ignore_actions_in_beginning_frames:
                if sensor_name.startswith('sensor.lidar'):
                    self.carla_queues["lidar_data_queue"].put(object_from_sensor)
                elif sensor_name.startswith('sensor.camera.rgb1'):
                    # In case of cam 1, store metadata
                    self.carla_queues["metadata_queue"].put(
                        CameraManager.get_metadata(weak_self, object_from_sensor.frame))
                    self.carla_queues["cam1_data_queue"].put(object_from_sensor)
                    print("Network provided new data!")
                elif sensor_name.startswith('sensor.camera.rgb2'):
                    self.carla_queues["cam2_data_queue"].put(object_from_sensor)
                elif sensor_name.startswith('sensor.camera.rgb3'):
                    self.carla_queues["cam3_data_queue"].put(object_from_sensor)

        # Display client
        if display_client_gui and self._sensors[self._shown_sensor_index][0].startswith(sensor_name.rstrip('0123456789')):
            if sensor_name.startswith('sensor.lidar'):
                points = np.frombuffer(object_from_sensor.raw_data, dtype=np.dtype('f4'))
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
                # x,y,z: +x is to the left of the car, +y is to the back of the car, +z is to the ground downwards.
                lidar_data = np.array(points[:, :2])
                lidar_data *= min(self._hud.dim) / 400.0  # TODO: denominator should be high for larger LiDAR ranges
                lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
                lidar_data = np.fabs(lidar_data)
                lidar_data = lidar_data.astype(np.int32)
                lidar_data = np.reshape(lidar_data, (-1, 2))
                # print number of scan points for debugging, it differs because of non reflections due to limited
                # range
                # print(len(lidar_data))
                lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
                lidar_img = np.zeros(lidar_img_size)
                lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
                self._surface = pygame.surfarray.make_surface(lidar_img)
            # skip showing left and right cameras
            # TODO: needs better handling as now backquote should be pressed 3 times to change viewed client camera
            elif sensor_name.startswith('sensor.camera.rgb2') or sensor_name.startswith('sensor.camera.rgb3'):
                return
            else:  # show front camera
                array = np.frombuffer(object_from_sensor.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (object_from_sensor.height, object_from_sensor.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
