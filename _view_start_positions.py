#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Connects with a CARLA simulator and displays the available start positions
for the current map."""

from __future__ import print_function

import argparse
import logging
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from adjustText import adjust_text

# To import carla
import os
import sys
try:  # Make sure to install CARLA 0.8.4 (prebuilt version)
    carla_path = os.environ['CARLA_PATH']
    sys.path.insert(0, carla_path + 'PythonClient')
except IndexError:
    pass
carla_path = os.environ['CARLA_PATH']

import subprocess, signal

from carla.client import make_carla_client
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
city_name = 'Town02'  # 'Town01' is IL training data, and 'Town02' is testing data


def view_start_positions(args):
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. The same way as in the client example.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        # We load the default settings to the client.
        scene = client.load_settings(CarlaSettings())
        print("Received the start positions")

        # We get the number of player starts, in order to detect the city.
        number_of_player_starts = len(scene.player_start_spots)
        if number_of_player_starts > 100:  # WARNING: unsafe way to check for city, see issue #313
            if sys.platform.startswith('win'):  # Windows OS
                image = mpimg.imread(carla_path + "PythonClient\carla\planner\Town01.png")
            else:  # Linux OS
                image = mpimg.imread(carla_path + "PythonClient/carla/planner/Town01.png")
            carla_map = CarlaMap('Town01', 0.1653, 50)
        else:
            if sys.platform.startswith('win'):  # Windows OS
                image = mpimg.imread(carla_path + "PythonClient\carla\planner\Town02.png")
            else:  # Linux OS
                image = mpimg.imread(carla_path + "PythonClient/carla/planner/Town02.png")
            carla_map = CarlaMap('Town02', 0.1653, 50)

        fig, ax = plt.subplots(1)

        ax.imshow(image)

        if args.positions == 'all':
            positions_to_plot = range(len(scene.player_start_spots))
        else:
            positions_to_plot = map(int, args.positions.split(','))

        texts = []
        for position in positions_to_plot:
            # Check if position is valid
            if position >= len(scene.player_start_spots):
                raise RuntimeError('Selected position is invalid')

            # Convert world to pixel coordinates
            pixel = carla_map.convert_to_pixel([scene.player_start_spots[position].location.x,
                                                scene.player_start_spots[position].location.y,
                                                scene.player_start_spots[position].location.z])

            circle = Circle((pixel[0], pixel[1]), 12, color='r', label='A point')
            ax.add_patch(circle)

            if not args.no_labels:
                texts.append(plt.text(pixel[0], pixel[1], str(position), size='x-small'))

        adjust_text(texts)

        plt.axis('off')
        plt.show()

        fig.savefig('town_positions.pdf', orientation='landscape', bbox_inches='tight')


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
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
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-pos', '--positions',
        metavar='P',
        default='all',
        help='Indices of the positions that you want to plot on the map. '
             'The indices must be separated by commas (default = all positions)')
    argparser.add_argument(
        '--no-labels',
        action='store_true',
        help='do not display position indices')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # Kill the server if already running
    if sys.platform.startswith('win'):  # Windows OS
        os.system("taskkill /f /im CarlaUE4.exe")
    else:  # Linux OS
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'CarlaUE4' in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)

    # Run Carla server
    init_time = 20.0
    if sys.platform.startswith('win'):  # Windows OS TODO: version 0.8.2 assume
        command = carla_path + 'CarlaUE4.exe /Game/Maps/' + city_name + ' -windowed -ResX=' + ' -world-port=' + \
                  str(args.port) + ' -carla-server-timeout=1000000000ms'
        print("CARLA server command called: " + command)
        FNULL = open(os.devnull, 'w')
        process = subprocess.Popen(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    else:  # Linux OS TODO: version 0.8.4 assumed
        bashCommand = carla_path + 'CarlaUE4.sh /Game/Maps/' + city_name + ' -windowed -ResX=' + ' -world-port=' + \
                      str(args.port) + ' -carla-server-timeout=1000000000ms'
        print("CARLA server command called: " + bashCommand)
        FNULL = open(os.devnull, 'w')
        process = subprocess.Popen(bashCommand, shell=True, stdout=FNULL, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
    time.sleep(init_time)  # Wait until CARLA is initialize

    while True:
        try:
            view_start_positions(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        except RuntimeError as error:
            logging.error(error)
            break


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
