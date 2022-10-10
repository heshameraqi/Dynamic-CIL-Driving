# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
import sys
import shutil  # For deleting data folder with all subfolders
import CarlaMiddleware
import socket

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
# Data is collected according to CARLA benchmark (Alexey Dosovitskiy, "CARLA: An Open Urban Driving Simulator")
#  only training weather conditions ([1, 3, 6, 8], test were [4, 14])
out_folder = '/media/cai1/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00'
metadata_filename = 'metadata.csv'
show_carla_server_gui = False
town = '01'  # '01' till '05', in Conditional Imitation Learning training dataset was from Town01
machine_ip = '10.67.84.8'
host = '127.0.0.1'
gpu_id = 0  # GPU to run simulator on
# game_fps:
# - Server game_fps=10 for the simulator means each consecutive displayed frames have 1/10 second of game time between
#   them.
# - The frequency of displaying these frames (simulator actual FPS) will be based on PC specs: Server FPS
# - With a camera and LiDAR, and ~40 vehicles, server FPS becomes from 7 to 20 (15 on average) on Titan Xp GPU PC
#   on Epic graphics and moderate resolution
# - Simulation time is the game time, the is the game world real-time. If very fast GPU in 1 second on the machine, the
#   game may execute 10 seconds of simulation (game time)
game_fps = 5
sim_width = 1280  # 1280 or 640 or 200, useless unless show_carla_server_gui=True and we want to see it in good graphics
sim_height = 720  # 720 or 480 or 88, useless unless show_carla_server_gui=True and we want to see it in good graphics
graphics_mode = 'Epic'  # 'Epic' or 'Low'
nVehicles = 20  # 60
# TODO: wait until walkers autopilot feature is supported by CARLA
drop_pedestrians = True  # This check is to decide whether to drop pedestrians or not
nPedestrians = 80  # This is the number of needed pedestrians (not guaranteed)  # TODO: make it guaranteed if needed?

# - If show_carla_clients_gui=True you should have X-server connection with the remote machine
#   (Having MobaXTerm connected in parallel with X11-Forwarding option enabled) (TODO)
# - Client HUD shows server FPS
# - frameNbr is increased each tick inside the Data collection infinite loop in CarlaMiddleware.py. Roughly, it's
#   increased each server frame because of an if condition waiting for the server
show_carla_clients_gui = True
show_hud = False
reset_if_zeroSpeed_for_time = 120  # 120, in seconds of game time (real world game time, not machine time)
# 30, in seconds of game time at the start of each episode (real world game time, not machine time). So simulator
#   environment is stable, for example no cars still being spawned
recording = True
# 3*60*60 means 3 hours of game time (driving simulation time, not machine time). Make sure that HDD don't run out of
#   memory
record_data_for_time = 8 * 60 * 60
# recording_each_frameNbr:
# 1*game_fps means each 1 second (client time: while true loop with if condition that waits the simulator, hence it's
#   equal to game time)
recording_each_frameNbr = (1. / 1.) * game_fps  # (1. / 5.) * game_fps saves 5 images per second
# Set to 20*60*game_fps to change weather each 20 minutes (simulation time, not machine time)
change_weather_each_frameNbr = 20 * 60 * game_fps
ignore_saving_in_beginning_seconds = 20  # 30
client_width = 180 * 1  # 180 or 640, this is also the resolution of the camera that is used to save data
client_height = 160 * 1  # 180 or 480, this is also the resolution of the camera that is used to save data

ego_car_mode = "autopilot"  # "autopilot" "roaming" "destination_pid", TODO: I changed agents modules so it doesn't work
destination = (10, 10, 0)  # if needed, based on ego_car_mode TODO: not used for now

# Get port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
s.listen(1)
port = s.getsockname()[1]
s.close()


# port = 2000


# ------------------------------------------------------------------------------
# Data collection function
# ------------------------------------------------------------------------------
def collect_data(carla_mw, out_folder_episode):
    # Clear output folder and create a new foldr if the script is recalled with --nodelete flag (usually due to car
    # stopped for long time)
    if recording:
        if len(sys.argv) > 1 and sys.argv[1] == '--nodelete':
            print("Creating a new episode inside out_folder!")
            tmpDir = out_folder_episode
            cnt = 0
            while os.path.isdir(tmpDir):
                tmpDir = out_folder + "/%i" % cnt
                cnt += 1
            out_folder_episode = tmpDir
        elif os.path.isdir(out_folder):
            print("Deleting old out_folder content!")
            shutil.rmtree(out_folder, ignore_errors=True)
        if not os.path.exists(out_folder_episode):
            os.makedirs(out_folder_episode)

    # Start CARLA simulator
    carla_mw.start_CARLA_simlator_ssh(town=town, show_carla_server_gui=show_carla_server_gui)

    # Reset CARLA simulator
    carla_mw.reset_simulator()

    # Spawn vehicles and pedestrians
    if drop_pedestrians:
        carla_mw.spawn_pedestrians(nPedestrians=nPedestrians)
    carla_mw.spawn_vehicles(nVehicles=nVehicles)

    # Create ego-car with sensors and attach Camera & LiDAR sensors to collect data
    carla_mw.create_move_ego_vehicle_w_sensors(mode=ego_car_mode, recording=recording,
                                               out_folder_episode=out_folder_episode,
                                               record_data_for_time=record_data_for_time)


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Create and initialize carla_mw object , make sure you have CARLA_0.9.3 prebuilt installed besides the project
    # directory (CARLA_0.9.3/CarlaUE4.sh is beside this project folder)
    carla_mw = CarlaMiddleware.CarlaMiddleware(game_fps=game_fps, host=host, port=port, gpu_id=gpu_id,
                                               metadata_filename=metadata_filename,
                                               show_carla_clients_gui=show_carla_clients_gui,
                                               recording_each_frameNbr=recording_each_frameNbr,
                                               change_weather_each_frameNbr=change_weather_each_frameNbr,
                                               sim_width=sim_width, sim_height=sim_height,
                                               client_width=client_width, client_height=client_height,
                                               reset_if_zeroSpeed_for_time=reset_if_zeroSpeed_for_time,
                                               ignore_saving_in_beginning_seconds=ignore_saving_in_beginning_seconds,
                                               show_hud=show_hud,
                                               graphics_mode=graphics_mode)  # TODO: If show_carla_server_gui=True, you can't run this code from SSH

    collect_data(carla_mw, out_folder_episode=out_folder + '/0')

    # Save Map
    # carla_mw.map.save_to_disk(path='output/whole_map_%s' % carla_mw.map.name)  # Save the map for opendrive format (tools/odrViewer64.1.9.1 to view the resulted xodr file)
