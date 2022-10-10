#TODO: AUC Data loader go from ['Follow-Lane', 'Go Straight', 'Go Right', 'Go Left'] to
# ['Go Right', 'Go Straight', 'Follow Lane', 'Go Left']

import os
import shutil

import numpy as np
import csv
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3d projection
import imageio
from tqdm import tqdm
import subprocess
import shutil  # For deleting data folder with all subfolders


# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
data_folder = '/media/cai1/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00/train'

'''
within a circle of this radius centered at intersection points in the map, the car is considered in the intersection,
and based on steering angle it the high level command becomes: right, left, or straight, not follow-lane
'''
radius_squared = 30**2  # 50**2
turn_max_threshold = 15  # in intersections, this decides going right or left

# Stopping detection
stopped_speed_threshold = 0.005  # in m/s
fps = 15  # TODO: should be automatically equal to the value at _collect_data.py time
interval_seconds = 2


# ------------------------------------------------------------------------------
# Create High Level Commands function
# ------------------------------------------------------------------------------
def inside_intersect(intersect, x, y):
    if ((x - intersect[0]) ** 2 + (y - intersect[1]) ** 2) < radius_squared:
        return True
    else:
        return False


def create_high_lvl_cmds(save_results_video=False, visualize=False):
    town1_map_intersect = [(90.923, 0.029), (155.874, 0.029), (336.683, 0.029),
                           (90.923, 57.529), (155.874, 57.529), (336.683, 57.529),
                           (90.923, 130.837), (336.683, 130.837),
                           (90.923, 197.593), (336.683, 197.593),
                           (90.923, 328.542), (336.683, 328.542),
                           ]

    # Parse x, y, and steering wheel angles
    episode = 0
    current_episode_dir = data_folder + "/%i" % episode  
    while os.path.isdir(current_episode_dir):
        # read input file
        input_csv_file = current_episode_dir + '/metadata.csv'
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
                    
        high_lvl_cmd_list = []
        os.remove(input_csv_file.replace('metadata.csv', 'metadata_with_highlvlcmd.csv'))  # remove if exists
        output_csv_file = open(input_csv_file.replace('metadata.csv', 'metadata_with_highlvlcmd.csv'), 'w+')
        output_csv_file_writer = csv.writer(output_csv_file, delimiter=',')
        output_csv_file_writer.writerow(
            ['step', 'steer', 'throttle', 'brake', 'forward_speed', 'yaw', 'location_x', 'location_y', 'high_lvl_cmd'])
        # high_lvl_cmds: 0 Right, 1 Straight, 2 Follow lane, 3 Left
        i = 0
        while i < len(x_list):
            x_inside = []
            y_inside = []
            # If within intersection
            for intersect in town1_map_intersect:
                while inside_intersect(intersect, x_list[i], y_list[i]):
                    x_inside.append(x_list[i])
                    y_inside.append(y_list[i])
                    i += 1
                    if i >= len(x_list):
                        break
                if x_inside:
                    break
            if x_inside:
                if (abs(x_inside[0]-x_inside[-1]) > turn_max_threshold) and \
                        (abs(y_inside[0]-y_inside[-1]) > turn_max_threshold):
                    if ((x_inside[-1] - x_inside[0]) * (y_inside[int(len(y_inside)/2)] - y_inside[0]) -
                        (y_inside[-1] - y_inside[0]) * (x_inside[int(len(x_inside)/2)] - x_inside[0])) >= 0:
                        cmd = 3  # Left
                    else:
                        cmd = 0  # Right
                else:
                    cmd = 1  # Straight
                for j in range(len(x_inside)):
                    high_lvl_cmd_list.append(cmd)
            else:
                high_lvl_cmd_list.append(2)  # Follow Lane
                i += 1

        # Remove zero speeds
        interval = fps * interval_seconds
        i = interval
        while i <= len(speeds_list):
            if all(s <= stopped_speed_threshold for s in speeds_list[i-interval:i]):
                high_lvl_cmd_list[i-1] = -1  # indication for stopping for while (needs to be processed in Data.py)
            i += 1

        # Add high_lvl_cmd_list information to metadata CSV file
        for i in range(len(high_lvl_cmd_list)):
            file_lines[i].append(high_lvl_cmd_list[i])
        for i in range(len(file_lines)):
            output_csv_file_writer.writerow(file_lines[i])
            output_csv_file.flush()

        print('Episode: ' + str(episode))
        print('  Number of datasamples: ' + str(len(high_lvl_cmd_list)))
        print('  Number of Right commands: ' + str(len([i for i in high_lvl_cmd_list if i == 0])))
        print('  Number of Straight commands: ' + str(len([i for i in high_lvl_cmd_list if i == 1])))
        print('  Number of Follow-lane commands: ' + str(len([i for i in high_lvl_cmd_list if i == 2])))
        print('  Number of Left commands: ' + str(len([i for i in high_lvl_cmd_list if i == 3])))
        print('  Number of Stopped samples: ' + str(len([i for i in high_lvl_cmd_list if i == -1])))
            
        # Visualization for debugging
        visualize = False
        nbrOfCommands = 6000  # 6000 or -1 for all
        if visualize:
            map_height = 600
            color_dict = {
                -1: "red",  # stopped
                0: "green",  # Right
                1: "black",  # Straight
                2: "yellow",  # Follow Lane
                3: "blue",  # Left
            }

            import pdb, sys
            cmds = high_lvl_cmd_list[0:nbrOfCommands]
            try:  # To import carla
                carla_path = os.environ['CARLA_PATH']
                sys.path.insert(0, carla_path + 'PythonClient')
            except IndexError:
                pass
            from carla.planner.map import CarlaMap
            import matplotlib.pyplot as plt

            plt.figure()
            map = CarlaMap('Town01', 0.1643, 50.0)
            map_size = map.map_image.shape
            map_image = map.get_map(map_height)
            plt.imshow(map_image)

            x_pos = []
            y_pos = []
            for i in range(len(cmds)):
                pos = map.convert_to_pixel([x_list[i], y_list[i], 0])
                y_pos.append(pos[1] * float(map_image.shape[1]) / float(map_size[1]))
                x_pos.append(pos[0] * float(map_image.shape[0]) / float(map_size[0]))
                if i == 0:
                    y_pos_start = pos[1] * float(map_image.shape[1]) / float(map_size[1])
                    x_pos_start = pos[0] * float(map_image.shape[0]) / float(map_size[0])
            plt.scatter(x_pos, y_pos, s=1, color=[color_dict[i] for i in cmds])
            plt.scatter([x_pos[i] for i in range(len(x_pos)) if high_lvl_cmd_list[i]==-1], 
                        [y_pos[i] for i in range(len(y_pos)) if high_lvl_cmd_list[i]==-1], 
                        s=20, marker='x', color=color_dict[-1])
            plt.scatter(x_pos_start, y_pos_start, s=40, facecolors='none', edgecolors='black')
            
            plt.xlabel('meters')
            plt.ylabel('meters')
			
            plt.show()
            #plt.show(block=False)

            #plt.figure()
            #plt.plot(high_lvl_cmd_list)
            #plt.show()

            # plt.show(block=False)
            # pdb.set_trace()

        # Select next episode folder
        episode += 1
        current_episode_dir = data_folder + "/%i" % episode


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating high level commands ...')
    create_high_lvl_cmds()
