# Adapted from:
# https://github.com/carla-simulator/imitation-learning/blob/master/run_CIL.py

import argparse
import logging
import sys
import os
import time
import subprocess, signal
import socket

# To import carla
try:  # Make sure to install CARLA 0.8.2 (prebuilt version)
    carla_path = os.environ['CARLA_PATH']
    sys.path.insert(0, carla_path + 'PythonClient')
except IndexError:
    pass

try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError('cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

from driving_benchmark import run_driving_benchmark
from imitation_learning import ImitationLearning
from carla.driving_benchmark.experiment_suites import CoRL2017


"""corl_2017 experiment suite: In each town (Town01 is training and Town02 is testing) there are 24 Experiments:
4 Modes (straight, one curve, navigation, navigation with people and other cars), this is repeated 6 different
weathers (train_weathers + test_weathers).
Each Experiment has 25 repetitions to try (source -> destination, pose pairs)."""

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
load_ready_model = False  # False, this is just because ready model assumed different order of network output branches
# model_path = '/media/cai1/data/heraqi/int-end-to-end-ad/models/IL/IL_ready/'
model_path = '/media/cai1/data/heraqi/int-end-to-end-ad/models/DED37A/epoch_18'
# model_path = '/media/cai1/data/heraqi/int-end-to-end-ad/models/0042E1_1/epoch_23'  # good model: epoch_23
town = 'Town01'  # 'Town01' is IL training data, and 'Town02' is testing data
gpu_id = 0  # GPU for running the the model, from 0 to 7 in DGX-1 server
model_inputs_mode =  "1cam" # "1cam" or "3cams" or "3cams-pgm"
kill_running_carla_servers = True  # False

show_carla_server_gui = True  # False
show_debug_info = False
if show_carla_server_gui:
    show_debug_info = True

game_fps = 15
sim_width = 1280
sim_height = 720
graphics_mode = 'Epic'  # 'Epic' or 'Low', should be Epic because training data was collected on epic
init_time = 20.0  # 40  # Waiting for CARLA server to initialize after starting

# Get port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
s.listen(1)
port = s.getsockname()[1]
s.close()
# port = 2000

if (__name__ == '__main__'):
    argparser = argparse.ArgumentParser(description=__doc__)
    # store_false to include messages added using logging.debug(), store_true to only show logging.info() messages
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
        '-c', '--city-name',
        metavar='C',
        default=town,
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the scripts'
    )
    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help='Uses the speed prediction branch to avoid unwanted agent stops'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the given log name'
    )
    args = argparser.parse_args()

    # Logger
    log_level = logging.DEBUG if (args.debug or show_debug_info) else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    # Kill the server if already running
    if kill_running_carla_servers:
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'CarlaUE4' in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)

    # Run Carla server
    bashCommand = carla_path + 'CarlaUE4.sh /Game/Maps/' + args.city_name + \
                  ' -windowed -ResX=' + str(sim_width) + ' -ResY=' + str(sim_height) + ' -world-port=' + \
                  str(args.port) + ' -timeout=100000000ms' + \
                  ' -carla-server -benchmark -fps=' + str(game_fps) + '-quality-level=' + graphics_mode
    # bashCommand = 'vglrun -d :7.' + str(gpu_id) + ' ' + bashCommand
    # If headless start of the simulator (no real window; headless) 
    if (not show_carla_server_gui):
        # setting the environment variable DISPLAY to empty
        bashCommand = 'DISPLAY= ' + bashCommand
    print("CARLA server command called: " + bashCommand)
    # Command: DISPLAY= /home/heraqi/CARLA_0.8.2/CarlaUE4.sh /Game/Maps/Town01 -windowed -ResX=1280 
    #   -ResY=720 -carla-port=2000 -timeout=100000000ms -carla-server -benchmark -fps=15-quality-level=Epic
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(bashCommand, shell=True, stdout=FNULL, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
    # output, error = process.communicate()
    # print('STDOUT:{}'.format(output))
    time.sleep(init_time)  # Wait until CARLA is initialized

    # Run benchmark as was in run_CIL.py
    agent = ImitationLearning(model_inputs_mode, model_path, args.avoid_stopping, load_ready_model=load_ready_model, gpu=gpu_id)
    corl = CoRL2017(args.city_name)
    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, corl, args.city_name, args.log_name, args.continue_experiment, args.host, args.port)
