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
from scipy.signal import medfilt


# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
data_folder = '/media/cai1/data/heraqi/int-end-to-end-ad/auc.carla.dataset'
town = '01'  # '01' or '03', ... this is needed for checking intersection hard-coded locations

'''
within a circle of this radius centered at intersection points in the map, the car is considered in the intersection,
and based on steering angle it the high level command becomes: right, left, or straight, not follow-lane
'''
radius_squared = 60**2  # 50**2
steer_threshold = 0.06  # in intersections above this threshold means going right or left
branches = ['Follow-Lane', 'Go Straight', 'Go Right', 'Go Left']  # command 0 means follow-lane, 1 means straight, ...
pgm_horizontal_angle_keep_ratio = 0.5  # if 0.5, 0.25 from the left and from the right of the PGM are not calculated


# ------------------------------------------------------------------------------
# Global variables
# ------------------------------------------------------------------------------
click_loc = None


# ------------------------------------------------------------------------------
# Cartesian to Polar coordinates conversion function
# ------------------------------------------------------------------------------
def cart2pol(x, y, z):
    xy = x**2 + y**2
    rho = np.sqrt(xy + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(xy))  # np.arctan2 retruns from -np.pi to np.pi

    # make angles from 0 to 360
    theta_deg = (np.degrees(theta) + 360) % 360
    phi_deg = (np.degrees(phi) + 360) % 360

    return rho, theta_deg, phi_deg


# ------------------------------------------------------------------------------
# Visualize collected data function
# ------------------------------------------------------------------------------
def generate_lidar_views(save_results_video=False, visualize=False):
    episode = 0
    current_episode_dir = data_folder + "/%i" % episode
    while os.path.isdir(current_episode_dir):
        # Todo:the sensor_name value in _parse_image call should be the same name as in automatic_control.py:
        #  create_sensors
        files = glob.glob(current_episode_dir + "/sensor.lidar.ray_cast/*")

        # Make 2 directories for LiDAR OGM and PGM
        folder_ogm = current_episode_dir + "/sensor.lidar.ray_cast.ogm"
        if not os.path.exists(folder_ogm):
            os.makedirs(folder_ogm)
        folder_pgm = current_episode_dir + "/sensor.lidar.ray_cast.pgm"
        if not os.path.exists(folder_pgm):
            os.makedirs(folder_pgm)

        # For each full scan
        if save_results_video or visualize:
            fig, ax = plt.subplots(2, 2)
        for f in tqdm(files):

            # if pgm file exists already don't re-create it
            # if os.path.isfile((folder_pgm + '/' + os.path.basename(f.name)).replace('.ply', '.jpg')):
            #     continue

            # Specific file for debugging
            # f = '/mnt/7A0C2F9B0C2F5185/heraqi/int-end-to-end-ad/auc.carla.dataset00/0/sensor.lidar/00007696.ply'

            with open(f) as f:
                content = f.readlines()
            x = np.zeros(len(content))
            y = np.zeros(len(content))
            z = np.zeros(len(content))
            for i in range(7, len(content)):
                row = content[i].split()
                y[i] = float(row[0])
                x[i] = float(row[1])
                z[i] = -float(row[2])

            # create OGM
            width = 500
            height = 500
            lidar_ogm = np.zeros((width, height), dtype=np.uint8)
            pos = np.vstack(([i+width/2. for i in x], [i+height/2. for i in y])).astype(int)
            lidar_ogm[tuple(pos)] = 255

            # create PGM
            # TODO: the following 3 parameters should be passed from outside to _collect_data.py and for this module
            nbrLayers = 32
            upper_fov = 10  # in degrees
            lower_fov = -30  # in degrees
            unreflected_value = 160  # in meters
            horizontal_angle_step = 3  # in degrees, should be divisible by 360

            # phi and theta deviations for a scan point to be associated to a cell, in degrees
            phi_half_cell_step = (upper_fov-lower_fov)/(2*nbrLayers)
            theta_half_cell_step = horizontal_angle_step/2.
            phi_ranges = np.arange(upper_fov, lower_fov, -(upper_fov - lower_fov)/nbrLayers) - phi_half_cell_step
            phi_ranges = (phi_ranges + 360) % 360
            horiz_start = int(360 * pgm_horizontal_angle_keep_ratio/2)
            theta_ranges = np.arange(horiz_start, 360-horiz_start, horizontal_angle_step) + theta_half_cell_step
            (rho, theta_deg, phi_deg) = cart2pol(x, y, z)

            lidar_pgm = np.zeros((nbrLayers, len(theta_ranges)), dtype=np.uint8)
            for i in range(lidar_pgm.shape[0]):
                for j in range(lidar_pgm.shape[1]):
                    # select points in that layer
                    # np.count_nonzero(indices_phi == True) should be zeros in case that this layer doesn't get
                    #   any reflections back in the sensor range
                    indices_phi = np.abs(phi_deg-phi_ranges[i]) <= 2*phi_half_cell_step  # TODO: multiple threshold value
                    # select horizontal sector
                    # np.count_nonzero(indices_theta == True) should be zeros in case that this layer doesn't get
                    #   any reflections back in the sensor range
                    indices_theta = np.abs(theta_deg-theta_ranges[j]) <= 2*theta_half_cell_step # TODO: multiple threshold value
                    rhos = rho[indices_phi & indices_theta]
                    # indices_phi & indices_theta
                    if len(rhos) > 0:
                        lidar_pgm[i, -j] = np.mean(rhos)
                    else:
                        lidar_pgm[i, -j] = unreflected_value

            # use log scale ?
            # lidar_pgm = np.log(lidar_pgm) / np.log(unreflected_value)

            # visualize for debugging
            folder_debug = current_episode_dir + "/sensor.lidar.debug"
            if save_results_video or visualize:
                if os.path.exists(folder_debug):
                    print("Deleting debug folder out_folder content!")
                    shutil.rmtree(folder_debug, ignore_errors=True)
                if not os.path.exists(folder_debug):
                    os.makedirs(folder_debug)

                ax[0, 0].imshow(lidar_ogm)
                # ax[0, 1].imshow((lidar_pgm*255.).astype(np.uint8))
                ax[0, 1].imshow(lidar_pgm)
                # ax[0, 1].imshow(lidar_pgm, cmap='hot', interpolation='nearest')

                import matplotlib.image as mpimage
                img_filename = f.name.replace("sensor.lidar", "sensor.camera.rgb1").replace(".ply", ".png")
                img = mpimage.imread(img_filename)
                ax[1, 0].imshow(img)

                # plt.show(block=False)
                fig.savefig((folder_debug + '/' + os.path.basename(f.name)).replace('.ply','.jpg'))
                if visualize:
                    plt.pause(0.001)
                    plt.draw()

                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(x, y, z)
                # ax.set_aspect('equal')  # Doesn't work yet

            # Save OGM and PGM
            imageio.imwrite((folder_ogm + '/' + os.path.basename(f.name)).replace('.ply', '.jpg'), lidar_ogm)
            imageio.imwrite((folder_pgm + '/' + os.path.basename(f.name)).replace('.ply', '.jpg'), lidar_pgm)

        # Saving video file
        if save_results_video:
            bashCommand = "cd " + folder_debug + "; ffmpeg -pattern_type glob -i \"*.jpg\" -c:v libx264 -profile:v " \
                                             "high -crf 20 -pix_fmt yuv420p _output.mp4"
            process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
            print("Video file saved")

        # Select next episode folder
        episode += 1
        current_episode_dir = data_folder + "/%i" % episode


# ------------------------------------------------------------------------------
# Create High Level Commands function
# ------------------------------------------------------------------------------
def onclick(event):  # to make the figure interacting and jump to time by clicking (if visualzie=True)
    global click_loc
    click_loc = event.xdata


def create_high_lvl_cmds(save_results_video=False, visualize=False):
    ''' Handle high level command; 1 for FOLLOW LANE, 2 for STRAIGHT, 3 for RIGHT, and 4 for LEFT
    Map GPS coordinates of intersections or roundabouts: to manually check the intersection locations in map:
    /home/heraqi/scripts/int-end-to-end-ad/int-end-to-end-ad-carla/_tools/odrViewer64.1.9.1
    then map xodr file like this:
    /home/heraqi/scripts/int-end-to-end-ad/CARLA_0.9.4/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr
    '''
    town1_map_intersect = [(90.923, 0.029), (155.874, 0.029), (336.683, 0.029),
                           (90.923, -57.529), (155.874, -57.529), (336.683, -57.529),
                           (90.923, -130.837), (336.683, -130.837),
                           (90.923, -197.593), (336.683, -197.593),
                           (90.923, -328.542), (336.683, -328.542),
                           ]

    town3_map_intersect = [(83.888, 255.128), (157.684, 258.336), (4.791,197.938),(85.750,198.438),
                           (154.028, 197.771), (-82.782, 136.291), (4.721, 137.267), (84.091, 134.990),
                           (154.028, 132.713), (82.790, 74.812), (151.751, 73.511), (-225.909, 3.899),
                           (-146.539, 1.297), (-80.830, -1.306), (0.166, 0.321), (80.838, 1.297),
                           (151.425, 0.971), (238.603, -0.655), (-147.189, -59.207), (168.666, -63.761),
                           (236.326, -61.809), (-222.656, -105.073), (-146.213, -104.422), (-80.830, -135.324),
                           (-0.159, -132.071), (-55.458, -196.479), (1.142, -199.731)]



    # Parse x, y, and steering wheel angles
    episode = 0
    current_episode_dir = data_folder + "/%i" % episode
    while os.path.isdir(current_episode_dir):
        read_csv_file_path = current_episode_dir + '/metadata.csv'
        write_csv_file_path = current_episode_dir + '/preprocessed_metadata.csv'
        start = 0
        end = 0
        direction = 0
        file_lines = []
        x_list = []
        y_list = []
        steer_list = []
        speeds_list = []
        with open(read_csv_file_path, encoding='utf-8') as csvfile:
            readfile = csv.reader(csvfile, delimiter=',')
            for row in readfile:
                if row[0] == 'data_file_name':  # ignore header
                    continue
                file_lines.append(row)
                x, y = row[3].replace(' ', '').replace(',', ' ').replace(')', '').replace('(', '').split()
                x_list.append(float(x))
                y_list.append(float(y))
                steer_list.append(float(row[6]))
                speeds_list.append(float(row[1]))

        if town == '03':
            map_intersect = town3_map_intersect
        elif town == '01':
            map_intersect = town1_map_intersect
        else:
            raise ValueError('Town intersection points not handled properly!')

        # Create high_lvl_cmd_list
        high_lvl_cmd_list = []
        for i in tqdm(range(len(x_list))):
            cmd = 1  # Follow Lane
            # If within intersection and brake is not zero
            if speeds_list[i] > 0:
                for j in range(len(map_intersect)):
                    if ((x_list[i] - map_intersect[j][0])**2 + (y_list[i] - map_intersect[j][1])**2) < \
                            radius_squared:
                        if float(steer_list[i]) > steer_threshold and float(steer_list[i]) != 0.0:
                            cmd = 3  # Right
                        elif float(steer_list[i]) < -steer_threshold and float(steer_list[i]) != 0.0:
                            cmd = 4  # Left
                        elif float(steer_list[i]) != 0.0:
                            cmd = 2  # Straight
                        break
            high_lvl_cmd_list.append(cmd)

        # Smooth the high level command
        high_lvl_cmd_list = medfilt([int(i) for i in high_lvl_cmd_list], 75)  # window size should be odd
        high_lvl_cmd_list = [int(i) for i in high_lvl_cmd_list]

        # Visualize high level command
        folder_debug = current_episode_dir + "/sensor.high_lvl_cmd.debug"
        if save_results_video or visualize:
            if os.path.exists(folder_debug):
                print("Deleting debug folder out_folder content!")
                shutil.rmtree(folder_debug, ignore_errors=True)
            if not os.path.exists(folder_debug):
                os.makedirs(folder_debug)

            global click_loc
            fig, ax = plt.subplots(2, 1)
            plt.subplot(2, 1, 1)
            plt.plot(high_lvl_cmd_list)
            fig_line = plt.axvline(x=i, color='r')
            ax[0].set_yticklabels(branches)
            plt.subplot(2, 1, 2)
            images = sorted(os.listdir(current_episode_dir + '/sensor.camera.rgb1/'))
            fig_text = plt.figtext(0.1, 0.1, "", backgroundcolor='w')
            img = plt.imread(current_episode_dir + '/sensor.camera.rgb1/' + images[i])
            fig_img = plt.imshow(img)
            if visualize:
                plt.show(block=False)
            i = 1
            while i < np.min([len(images), 4000]):
                print(i)
                img = plt.imread(current_episode_dir + '/sensor.camera.rgb1/' + images[i])
                fig_img.set_data(img)
                fig_line.set_xdata(i)
                fig_text.set_text("Steer={0:0.4f}, X={1:.0f}, Y={2:.0f}\n"
                                  "File name:{3:s}\nCommand={4:s}".format(steer_list[i], x_list[i], y_list[i],
                                                                          images[i], branches[high_lvl_cmd_list[i]-1]))
                if visualize:
                    fig.canvas.mpl_connect('button_press_event', onclick)
                    plt.pause(0.001)
                    plt.draw()
                if click_loc is None:
                    i += 5
                else:
                    i = int(click_loc)
                    click_loc = None
                fig.savefig(folder_debug + '/' + images[i])
            # Saving video file
            if save_results_video:
                bashCommand = "cd " + folder_debug + "; ffmpeg -pattern_type glob -i \"*.png\" -c:v libx264 -profile:v " \
                                                     "high -crf 20 -pix_fmt yuv420p _output.mp4"
                process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
                print("Video file savdeo file saved")

        # TODO: Look ahead and modify high_lvl_cmd_list, it's like filtering the high level commands


        # Add high_lvl_cmd_list information to metadata CSV file
        for i in range(len(high_lvl_cmd_list)):
            file_lines[i].append(high_lvl_cmd_list[i])
        with open(write_csv_file_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(
                ['data_file_name', 'speed_kmh', 'heading', 'location', 'GNSS', 'throttle', 'steer', 'brake', 'reverse',
                 'hand_brake', 'manual', 'gear', 'collision', 'nbr_vehicles', 'weather', 'high_lvl_cmd'])
            for i in range(len(file_lines)):
                csv_writer.writerow(file_lines[i])

        # Select next episode folder
        episode += 1
        current_episode_dir = data_folder + "/%i" % episode


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating high level commands ...')
    create_high_lvl_cmds()
    print('Generate LiDAR views (OGM\'s and PGM\'s ...')
    generate_lidar_views()
