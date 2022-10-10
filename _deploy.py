# Adapted from:draw_ogm_map
# https://github.com/carla-simulator/imitation-learning/blob/master/run_CIL.py
"""corl_2017 experiment suite: In each town (Town01 is training and Town02 is testing) there are 24 Experiments:
4 Modes (straight, one curve, navigation, navigation with people and other cars), this is repeated 6 different
weathers (train_weathers + test_weathers).
Each Experiment has 25 repetitions to try (source -> destination, pose pairs)."""

import argparse
import logging
import sys
import os
import time
import subprocess, signal
import socket

# To import carla
try:  # Make sure to install CARLA 0.8.4 (prebuilt version)
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
# from carla.driving_benchmark.experiment_suites import CoRL2017  # CoRL2017 experiment

# ------------------------------------------------------------------------------
# Initializations
# ------------------------------------------------------------------------------
load_ready_model = False  # False, this is just because ready model assumed different order of network output branches
our_experiment_sensor_setup = True  # True
gpu_memory_fraction = 0.2  # 0.25 for 1 CNN stream model, 0.4 for the bigger
prefix = 'AUCdata'  # experiment output folder name prefix ('AUCdata', 'AUCdata_WZ', 'AUCdata_WZ_NoRect', ...)

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
working_zones_experiments = False  # Use the experiment_working_zones experiment suite instead of experiment_corl17
continue_experiment = False
town = 'Town01'  # 'Town01' is IL training data, and 'Town02' is testing data
gpu_id = 0  # GPU for running the the model, from 0 to 7 in DGX-1 server
kill_running_carla_servers = True  # True
enable_WZ_avoidance_using_OGM = False  # True
enable_steering_rect_using_OGM = True  # True, if True enable_WZ_avoidance_using_OGM must also be True
start_from_exp_pose = (3, 12)  # (1,1) is the default, Numbers not indices. If continue_experiment=False or True on empty experiment. Interesting experiments: 2,2 for town 2 in working zones benchmark

# Planning visualization configurations
save_ogm_planner_figure = False  # False, save the plots
visualize_ogm_planner = False  # False, show the plots
start_visualize_or_save_from_frame = 0  # 0, start consuming time from that frame
normal_save_quality = 50  # 50 for normal quality, 300 or 400 for decent quality
save_high_quality_frame_numbers = []  # [434, 522, 864] on ex = 4 and pos = 9 CoRL benchamrk, list(range(500, 900))
visualize_save_directory = r"C:\\Work\\Software\\CARLA\\results\\ogm\\"

# Pre-trained CIL model on CIL dataset
'''model_path = '/media/heraqi/data/heraqi/int-end-to-end-ad/models/IL/IL_ready'
model_inputs_mode = "1cam"
our_experiment_sensor_setup = False
prefix = 'Pretrained'  # experiment output folder name prefix
load_ready_model = True'''

# Train CIL model on CIL dataset from scratch (deployed on fps=10)
'''# model_path = '/home/heraqi/scripts/models/F3F1E4_CIL_data/epoch_53'  # Model trained
model_path = '/mnt/sdb1/heraqi/data/int-end-to-end-ad/models/models/F3F1E4_CIL_data_3/epoch_24
model_inputs_mode = "1cam"
our_experiment_sensor_setup = False
prefix = 'CIL_data'  # experiment output folder name prefix'''

# Train single camera on our dataset
'''# model_path = '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F034BF_AUC2_data_1cam_26/epoch_17'
model_path = r"C:\Work\Software\CARLA\models\F034BF_AUC2_data_1cam_26\epoch_17"
model_inputs_mode = "1cam"'''

# Train single camera and LiDAR PGM on our dataset
model_path = r"C:\Work\Software\CARLA\models\F034BF_AUC2_data_1cam-pgm_4\epoch_22"
# model_path = '/mnt/sdb1/heraqi/data/int-end-to-end-ad/models/F034BF_AUC2_data_1cam-pgm_4/epoch_22'
model_inputs_mode = "1cam-pgm"

# Train 3 cameras on our dataset
'''model_path = '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F034BF_AUC2_data_3cams_4/epoch_11'  # trained on smaller data
model_inputs_mode = "3cams"'''

# Train 3 cameras and LiDAR PGM on our dataset
'''model_path = '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F034BF_AUC2_data_3cams-pgm_9/epoch_32' # trained on smaller data
gpu_memory_fraction = 1.0
model_inputs_mode = "3cams-pgm"'''

# Configurations that are rarely changed
game_fps = 15  # 15 (auc2 data) or 15 (auc1 data)
sim_width = 1280
sim_height = 720
graphics_mode = 'Epic'  # 'Epic' or 'Low', should be Epic because training data was collected on epic
init_time = 20.0  # 40  # Waiting for CARLA server to initialize after starting

# Intializations
if working_zones_experiments:
    from experiment_working_zones import experiment
else:
    from experiment_corl17 import experiment

show_carla_server_gui = False  # False
show_debug_info = False
if show_carla_server_gui:
    show_debug_info = True

# Get port
if sys.platform.startswith('win'):  # Windows OS
    port = 2002  # TODO: this works on Windows, what about Linux the old option was better?
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()

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
        '--avoid-stopping',
        default=True,
        action='store_false',
        help='Uses the speed prediction branch to avoid unwanted agent stops')
    args = argparser.parse_args()

    # Logger
    log_level = logging.DEBUG if (args.debug or show_debug_info) else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    # Kill the server if already running
    if kill_running_carla_servers:
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
    if sys.platform.startswith('win'):  # Windows OS TODO: version 0.8.2 assumed
        command = carla_path + 'CarlaUE4.exe /Game/Maps/' + args.city_name + \
                  ' -windowed -ResX=' + str(sim_width) + ' -ResY=' + str(sim_height) + ' -world-port=' + \
                  str(args.port) + ' -carla-server-timeout=1000000000ms' + \
                  ' -carla-server -benchmark -fps=' + str(game_fps) + ' -quality-level=' + graphics_mode
        # command = 'vglrun -d :7.' + str(gpu_id) + ' ' + command
        # If headless start of the simulator (no real window; headless)
        if (not show_carla_server_gui):
            # setting the environment variable DISPLAY to empty
            command = "start /min \"\" " + command + ""  # TODO: hide CARLA in Windows

        print("CARLA server command called: " + command)
        # Command: C:\Work\Software\CARLA\CARLA_0.8.2\CarlaUE4.exe /Game/Maps/Town02 -windowed -ResX=1280 -ResY=720
        # -world-port=2002 -carla-server-timeout=1000000000ms -carla-server -benchmark -fps=15
        FNULL = open(os.devnull, 'w')
        process = subprocess.Popen(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    else:  # Linux OS TODO: version 0.8.4 assumed
        bashCommand = carla_path + 'CarlaUE4.sh /Game/Maps/' + args.city_name + \
                      ' -windowed -ResX=' + str(sim_width) + ' -ResY=' + str(sim_height) + ' -world-port=' + \
                      str(args.port) + ' -carla-server-timeout=1000000000ms' + \
                      ' -carla-server -benchmark -fps=' + str(game_fps) + ' -quality-level=' + graphics_mode
        # bashCommand = 'vglrun -d :7.' + str(gpu_id) + ' ' + bashCommand
        # If headless start of the simulator (no real window; headless)
        if (not show_carla_server_gui):
            # setting the environment variable DISPLAY to empty
            bashCommand = 'DISPLAY= ' + bashCommand

        print("CARLA server command called: " + bashCommand)
        # Simple command: CarlaUE4.exe /Game/Maps/Town01 -ResX=1280 -ResY=720 -quality-level=Epic
        # Command: DISPLAY= /home/heraqi/CARLA_0.8.2/CarlaUE4.sh /Game/Maps/Town01 -windowed -ResX=1280
        #   -ResY=720 -carla-port=2000 -timeout=10000000ms -carla-server -benchmark -fps=15 -quality-level=Epic
        FNULL = open(os.devnull, 'w')
        process = subprocess.Popen(bashCommand, shell=True, stdout=FNULL, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
        # output, error = process.communicate()
        # print('STDOUT:{}'.format(output))
    time.sleep(init_time)  # Wait until CARLA is initialize

    # Run benchmark as was in run_CIL.py
    agent = ImitationLearning(model_inputs_mode, our_experiment_sensor_setup, model_path, args.avoid_stopping,
                              load_ready_model=load_ready_model, gpu=gpu_id, gpu_memory_fraction=gpu_memory_fraction)
    experiment = experiment(model_inputs_mode, our_experiment_sensor_setup, args.city_name)
    # Now actually run the driving_benchmark
    model_folder_name = os.path.basename(os.path.normpath(os.path.abspath(os.path.join(model_path, "../"))))
    epoch = os.path.basename(os.path.normpath(model_path))
    
    # Run the actual benchmark
    run_driving_benchmark(
        agent, experiment, args.city_name, prefix, model_folder_name + '_' + epoch, continue_experiment,
        start_from_exp_pose, args.host, args.port, enable_WZ_avoidance_using_OGM, enable_steering_rect_using_OGM,
        game_fps, visualize_ogm_planner=visualize_ogm_planner, save_ogm_planner_figure=save_ogm_planner_figure,
        start_visualize_or_save_from_frame=start_visualize_or_save_from_frame,
        normal_save_quality=normal_save_quality, save_high_quality_frame_numbers=save_high_quality_frame_numbers,
        visualize_save_directory=visualize_save_directory)
