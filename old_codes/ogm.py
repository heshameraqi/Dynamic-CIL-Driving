import numpy as np
import numpy.matlib as ml
import matplotlib.pyplot as plt
from matplotlib.path import Path
# from transform import *
import bisect
import math
import time
from datetime import datetime
# from image import load_image
from scipy.signal import savgol_filter
# from matplotlib import animation
from sympy import sympify
from sympy.geometry import Point2D, Segment2D, Circle
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely import geometry
from math import *  # Not so neat, but to make code more readable
import itertools
import imageio
import scipy
# from Oxford_robotcar_parser import read_oxford_data
# from OGM_VoC_Scala import read_scala_data
# For debugging:
import argparse
from pdb import set_trace  # For debugging
# from camera_model import CameraModel
import os
import re
# from transform import build_se3_transform
# from build_pointcloud import *


# ------------------------------------------------------------------------------
# Configurations class
# ------------------------------------------------------------------------------
class configuration:
    def __init__(
            self):  # is_training is passed to model_creator function, then to slim.arg_scope then to all slim.batch_norm and slim.dropout layers
        self.width = 100  # meters
        self.height = 100  # meters
        self.reslution = 0.2  # meters per grid cell. width and height should be mutliple of it
        self.beam_azimuth = 0.25 * np.pi / 180.0  # [radian] #beta
        self.wall_depth = 4  # [meters] #alpha
        self.range = 50 / self.reslution  # [meters] #300 for oxford, 50 for Scala
        self.prop_free = 0.35
        self.prop_occ = 0.65
        self.prop_noInfo = 0.5
        self.dx = 0.0
        self.dy = 0.0
        self.retrieve_time = 0
        self.update_time = 0
        self.affected_points = 0
        self.count_scans = 0
        self.object_reflectivity = 10
        self.car_x_fractions = 0
        self.car_y_fractions = 0
        self.p_local = np.array([0, 0, 0])
        self.log_prop_occ = np.log(self.prop_occ / self.prop_free)
        self.log_prop_free = np.log(self.prop_free / self.prop_occ)
        self.circle_r = 0 / self.reslution
        self.current_velocity = 0


class OGM():
    def __init__(self):
        # Load configurations
        self.cfg = configuration()
        # Create one-dimensional arrays for all x and y of map
        y = range((int)(self.cfg.height / self.cfg.reslution))
        x = range((int)(self.cfg.width / self.cfg.reslution))
        self.map_coordinates = np.array(list(itertools.product(x, y)))

        self.map_positions = np.array([np.tile(np.arange(0, (int)(self.cfg.height / self.cfg.reslution))[:, None],
                                               (1, (int)(self.cfg.width / self.cfg.reslution))),
                                       np.tile(np.arange(0, (int)(self.cfg.width / self.cfg.reslution))[:, None].T,
                                               ((int)(self.cfg.height / self.cfg.reslution), 1))])

        self.map = np.ones(((int)(self.cfg.height / self.cfg.reslution),
                            (int)(self.cfg.width / self.cfg.reslution))) * self.cfg.prop_noInfo
        self.log_odd_map = np.zeros(((int)(self.cfg.height / self.cfg.reslution), (int)(self.cfg.width / self.cfg.reslution)))

        self.old_pose = 0  # unset
        self.Initial_pose = np.array([250, 250, 0])
    # ------------------------------------------------------------------------------
    # OGM Local Functions
    # ------------------------------------------------------------------------------
    '''def update_ogm(self, scan, pose, old_pose, cfg, map_coordinates, map):  # pose = [y,x,yaw] in [pixels]
        # Poses from meters to pixels
        pose[0:2] = pose[0:2]
        old_pose[0:2] = old_pose[0:2]

        # Calculate Map Shift by x,y difference of pose and old_pose
        y_shift = int(pose[0] - old_pose[0])  # Down with non-fraction part
        x_shift = int(pose[1] - old_pose[1])  # Left with non-fraction part
        return_pose = old_pose.copy()
        return_pose[0] += ((pose[0] - old_pose[0]) - int(pose[0] - old_pose[0]))  # old_pose + fraction part
        return_pose[1] += ((pose[1] - old_pose[1]) - int(pose[1] - old_pose[1]))  # old_pose + fraction part
        return_pose[2] = pose[2].copy()

        # set_trace()

        # Do actual shifting
        map = np.roll(map, y_shift, axis=0)  # down
        map = np.roll(map, x_shift, axis=1)  # left
        if (y_shift > 0):
            map[0:y_shift, :] = cfg.prop_noInfo
        elif (y_shift < 0):
            map[y_shift:-1, :] = cfg.prop_noInfo
        if (x_shift > 0):
            map[:, -x_shift:-1] = cfg.prop_noInfo
        elif (x_shift < 0):
            map[:, 0:x_shift] = cfg.prop_noInfo

        # Get all angles and distances
        angles = np.zeros(len(scan))
        distances = np.zeros(len(scan))
        for i in range(len(scan)):
            angles[i] = atan2(scan[i][0], scan[i][1])
            distances[i] = sqrt(scan[i][0] ** 2 + scan[i][1] ** 2)

        # Get affected area convex hull
        scan = scan / cfg.reslution  # Normalize to pixels space
        scan[:, 1] += pose[1]  # Shift by car position
        scan[:, 0] = pose[0] - scan[:, 0]  # Shift by car position
        cloud = np.vstack([scan[:, :2], np.array([pose[0], pose[1]])])
        hull = geometry.Polygon([[p[0], [1]] for p in cloud])

        # hull = ConvexHull(cloud)  # Add LiDAR centre to convex
        hull_vertices = cloud[hull.vertices, :2]
        if not isinstance(hull_vertices, Delaunay):
            hull_del = Delaunay(hull_vertices)
            affected_map_coordinates = self.map_coordinates[hull_del.find_simplex(self.map_coordinates) >= 0]
        else:
            raise ValueError("Convex Hull Problem")

        ## Show affected area for debugging
        for p in affected_map_coordinates:
            if (p[0] > 0 and p[1] > 0 and p[0] < map.shape[0] and p[1] < map.shape[1]):
                map[(int)(p[0]), (int)(p[1])] = 0.3
        plt.figure()
        plt.imshow(map)
        plt.show(block=False)
        # set_trace()

        # Get new probabilities and update intensities
        new_probs = np.zeros(len(affected_map_coordinates))
        for i in range(len(affected_map_coordinates)):
            dist_car = sqrt((affected_map_coordinates[i][0] - pose[0]) ** 2 +
                            (affected_map_coordinates[i][1] - pose[1]) ** 2) * cfg.reslution
            angle = atan2(pose[0] - affected_map_coordinates[i][0], affected_map_coordinates[i][1] - pose[1])
            nearest_scanpoint_idx = (np.abs(angles - angle)).argmin()

            if (dist_car < (distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                new_probs[i] = cfg.prop_free
            elif (dist_car <= (distances[nearest_scanpoint_idx] + cfg.wall_depth / 2.)):
                new_probs[i] = cfg.prop_occ
            else:
                new_probs[i] = cfg.prop_noInfo

            # Do actual map update (Bayesian Filter)
        for i in range(len(affected_map_coordinates)):
            map[affected_map_coordinates[i][0], affected_map_coordinates[i][1]] = 1. / (
                    1. +
                    (1. - new_probs[i]) / new_probs[i] *  # Given Zt and Xt
                    (1. - map[affected_map_coordinates[i][0], affected_map_coordinates[i][1]]) /
                    map[affected_map_coordinates[i][0], affected_map_coordinates[i][1]] *  # Recursive Term
                    cfg.prop_noInfo / (1. - cfg.prop_noInfo)  # Perior Probability
            )

        return map, return_pose'''

    def rotate_points(self, origin, points, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        while (angle > np.pi):
            angle -= 2. * np.pi
        while (angle < -np.pi):
            angle += 2. * np.pi
        qx = []
        qy = []
        for point in points:
            ox = origin[0]
            oy = origin[1]
            px = point[0]
            py = point[1]
            # angle=point[2]

            # qx.append(ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy))
            # qy.append(oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy))
            qx.append(math.cos(angle) * px - math.sin(angle) * py)
            qy.append(math.sin(angle) * px + math.cos(angle) * py)
        return [qx, qy]

    def update_ogm(self, mode, scan, pose, Local_Pose, cfg, map_coordinate, map):
        x_shift = 0
        y_shift = 0
        if cfg.dx > 0 or cfg.dx < 0:
            cfg.car_x_fractions += cfg.dx
            cfg.dx = 0
        while cfg.car_x_fractions >= 1 or cfg.car_x_fractions <= -1:
            # x_shift = int(old_pose[0] - pose[0])  # Down with non-fraction part
            x_shift += int(cfg.car_x_fractions * -1)
            print('x_shift', x_shift)
            cfg.car_x_fractions = cfg.car_x_fractions - int(cfg.car_x_fractions)
            map = np.roll(map, x_shift, axis=1)  # left

        if cfg.dy < 0 or cfg.dy > 0:
            cfg.car_y_fractions += cfg.dy
            cfg.dy = 0
        while cfg.car_y_fractions >= 1 or cfg.car_y_fractions <= -1:
            # y_shift = int(old_pose[1] - pose[1])  # Left with non-fraction part
            y_shift += int(cfg.car_y_fractions)
            print('y_shift', y_shift)
            cfg.car_y_fractions = cfg.car_y_fractions - int(cfg.car_y_fractions)
            map = np.roll(map, y_shift, axis=0)  # down
        if (y_shift < 0):  # y_shift=-1, we need to reset the last row
            map[y_shift:, :] = cfg.prop_noInfo
        elif (y_shift > 0):  # y_shift=1, we need te reset the first row
            map[0:y_shift, :] = cfg.prop_noInfo
        if (x_shift < 0):  # x_shift=-1, we need to reset the last column
            map[:, x_shift:] = cfg.prop_noInfo
        elif (x_shift > 0):  # x_shift=1, we need to reset the first column
            map[:, 0:x_shift] = cfg.prop_noInfo

        # map = np.roll(map, x_shift, axis=1)  # down
        map = np.roll(map, self.Circle_shift[1], axis=0)  # right
        map = np.roll(map, self.Circle_shift[0], axis=1)  # down
        if (self.Circle_shift[1] < 0):  # y_shift=-1, we need to reset the last row
            map[self.Circle_shift[1]:, :] = cfg.prop_noInfo
        elif (self.Circle_shift[1] > 0):  # y_shift=1, we need te reset the first row
            map[0:self.Circle_shift[1], :] = cfg.prop_noInfo
        if (self.Circle_shift[0] < 0):  # x_shift=-1, we need to reset the last column
            map[:, self.Circle_shift[0]:] = cfg.prop_noInfo
        elif (self.Circle_shift[0] > 0):  # x_shift=1, we need to reset the first column
            map[:, 0:self.Circle_shift[0]] = cfg.prop_noInfo

        scan = scan / cfg.reslution  # Normalize to pixels space
        scan_copy = scan.copy()

        Actuall_Prob = map  # 1.0 - 1. / (1. + np.exp(map))
        # Get all angles and distances
        angles = np.zeros(len(scan))
        distances = np.zeros(len(scan))

        if x_shift >= 1 or x_shift <= -1 or y_shift >= 1 or y_shift <= -1 \
                or cfg.car_x_fractions > 0 or cfg.car_x_fractions < 0 or cfg.car_y_fractions > 0 or cfg.car_y_fractions < 0:
            if (x_shift > 0 or x_shift < 0 or cfg.car_x_fractions > 0 or cfg.car_x_fractions < 0):
                scan_rotated = self.rotate_points((0, 0), scan, pose[2])
                scan_rotated = np.asarray(scan_rotated).T
                scan_copy[:, 0] = Local_Pose[0] + scan_rotated[:, 0]  # +self.Circle_shift[0]  # Shift by car position
                scan_copy[:, 1] = Local_Pose[1] + scan_rotated[:, 1]  # +self.Circle_shift[1] # Shift by car position
            elif (y_shift > 0 or y_shift < 0 or cfg.car_y_fractions > 0 or cfg.car_y_fractions < 0):
                # if the change in Y is negative, we should + scan_rotated in X and Y
                scan_rotated = self.rotate_points((0, 0), scan, pose[2])
                scan_rotated = np.asarray(scan_rotated).T
                scan_copy[:, 0] = Local_Pose[0] + scan_rotated[:, 0]  # +self.Circle_shift[0] # Shift by car position
                scan_copy[:, 1] = Local_Pose[1] + scan_rotated[:, 1]  # +self.Circle_shift[1] # Shift by car position

            # Visualize & Save Scans
            # temp_map= Actuall_Prob.copy()
            # for p in scan_copy:
            #     if (p[0] > 0 and p[1] > 0 and p[0] < map.shape[0] and p[1] < map.shape[1]):
            #         temp_map[(int)(p[1]), (int)(p[0])] = 1
            # plt.imshow(temp_map, 'Greys')
            # plt.savefig('D:/Valeo/OGM/OGM_VOC_Scans/' + str(loop_idx) + '.png', dpi=700)
            # plt.show(block=False)
            cloud = np.vstack([scan_copy[:, :2], np.array([Local_Pose[0], Local_Pose[1]])])
            if mode == 'ConvexHull':
                start_time = time.time()
                C_hull = ConvexHull(cloud)  # Add LiDAR centre to convex
                hull_vertices = cloud[C_hull.vertices, :2]
                if not isinstance(hull_vertices, Delaunay):
                    hull_del = Delaunay(hull_vertices)
                    u = hull_del.find_simplex(map_coordinate)
                    affected_map_coordinates = map_coordinate[u >= 0]
                else:
                    raise ValueError("Convex Hull Problem")
                print("Convex Hull Time", str(time.time() - start_time))

            temp_map = Actuall_Prob.copy()
            for p in affected_map_coordinates:
                if (p[0] > 0 and p[1] > 0 and p[0] < map.shape[0] and p[1] < map.shape[1]):
                    temp_map[(int)(p[1]), (int)(p[0])] = 0.3
            plt.imshow(temp_map, 'Greys')
            # plt.savefig('D:/Valeo/OGM/OGM_VOC_Scans/' + str(scan_idx) + '.png', dpi=700)
            if mode == 'Polygon':
                tupVerts = tuple(cloud)
                min_scan_x = int(np.min(scan_copy[:, 0]))  # +init_pose[0]#+pose[0]
                max_scan_x = int(np.max(scan_copy[:, 0]))  # +init_pose[0]#+pose[0]
                min_scan_y = int(np.min(scan_copy[:, 1]))  # +init_pose[1]#+pose[1]
                max_scan_y = int(np.max(scan_copy[:, 1]))  # +init_pose[1]#+pose[1]
                start_time = time.time()
                x, y = np.meshgrid(np.arange(min_scan_x, max_scan_x),
                                   np.arange(min_scan_y, max_scan_y))  # make a canvas with coordinates
                x, y = x.flatten(), y.flatten()
                points = []
                points = np.vstack((x, y)).T
                p = Path(tupVerts)  # make a polygon
                grid = p.contains_points(points)
                shaped_mask = grid.reshape(points.shape[0], 1)
                affected_map_coordinates = []
                for i in range(shaped_mask.shape[0]):
                    if shaped_mask[i] == True:
                        affected_map_coordinates.append(np.vstack((points[i][0], points[i][1])).T)
                affected_map_coordinates = np.array(affected_map_coordinates).squeeze(axis=1)
                print("Polygon Time", str(time.time() - start_time))
            cfg.retrieve_time += time.time() - start_time
            start_time = time.time()
            for i in range(len(scan_copy)):
                distances[i] = sqrt(
                    (scan_copy[i][0] - Local_Pose[0]) ** 2 + (scan_copy[i][1] - Local_Pose[1]) ** 2)  # r
                angles[i] = atan2(scan_copy[i][1] - Local_Pose[1], scan_copy[i][0] - Local_Pose[0])  # b
                while (angles[i] > np.pi):
                    angles[i] -= 2. * np.pi
                while (angles[i] < -np.pi):
                    angles[i] += 2. * np.pi

            new_probs = np.zeros(len(affected_map_coordinates))
            angle_diff = np.zeros(len(affected_map_coordinates))

            guranteed_range = 50 / cfg.reslution
            for i in range(len(affected_map_coordinates)):
                dist_car = sqrt((affected_map_coordinates[i][0] - Local_Pose[0]) ** 2 +
                                (affected_map_coordinates[i][1] - Local_Pose[1]) ** 2)  # * cfg.reslution
                angle = atan2(affected_map_coordinates[i][1] - Local_Pose[1],
                              affected_map_coordinates[i][0] - Local_Pose[0])
                while (angle > np.pi):
                    angle -= 2. * np.pi
                while (angle < -np.pi):
                    angle += 2. * np.pi
                angle_difference = np.min(np.abs(angles - angle))
                angle_diff[i] = angle_difference
                nearest_scanpoint_idx = (np.abs(angles - angle)).argmin()
                if (np.abs(angle_difference) < cfg.beam_azimuth / 2):
                    if (dist_car < (distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                        new_probs[i] = cfg.prop_free
                    elif (dist_car <= (distances[nearest_scanpoint_idx] + cfg.wall_depth / 2.)):
                        new_probs[i] = cfg.prop_occ
                    else:
                        new_probs[i] = cfg.prop_noInfo
                else:
                    if ((dist_car <= guranteed_range) & (
                            dist_car < distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                        new_probs[i] = cfg.prop_free
                    elif ((dist_car > guranteed_range) & (
                            dist_car < distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                        new_probs[
                            i] = cfg.prop_free  # Oxford_linear_decay(init_pose[0]+guranteed_range, init_pose[0]+(cfg.range/cfg.reslution), init_pose[0]+(dist_car))
                    else:
                        new_probs[i] = cfg.prop_noInfo

            for i in range(len(affected_map_coordinates)):
                Actuall_Prob[affected_map_coordinates[i][1], affected_map_coordinates[i][0]] = 1. / (
                        1. +
                        (1. - new_probs[i]) / new_probs[i] *  # Given Zt and Xt
                        (1. - Actuall_Prob[affected_map_coordinates[i][1], affected_map_coordinates[i][0]]) /
                        Actuall_Prob[affected_map_coordinates[i][1], affected_map_coordinates[i][0]] *  # Recursive Term
                        cfg.prop_noInfo / (1. - cfg.prop_noInfo)  # Perior Probability
                )
            cfg.affected_points += len(affected_map_coordinates)
            cfg.update_time += time.time() - start_time
            cfg.count_scans += 1
            print("Convex Hull points = ", len(affected_map_coordinates), "Convex Hull Time update",
                  str(time.time() - start_time))
        #        print("Polygon points = ", len(affected_map_coordinates), "Total Polygon Time",
        #             str(time.time() - start_time))
        return Actuall_Prob, x_shift, y_shift

    def update_ogm_Log_odd(self, mode, scan, pose, Local_Pose, cfg, map_coordinate, log_odd_map):
        x_shift = 0
        y_shift = 0
        if cfg.dx > 0 or cfg.dx < 0:
            cfg.car_x_fractions += cfg.dx
            cfg.dx = 0
        while cfg.car_x_fractions >= 1 or cfg.car_x_fractions <= -1:
            # x_shift = int(old_pose[0] - pose[0])  # Down with non-fraction part
            x_shift += int(cfg.car_x_fractions * -1)
            print('x_shift', x_shift)
            cfg.car_x_fractions = cfg.car_x_fractions - int(cfg.car_x_fractions)
            log_odd_map = np.roll(log_odd_map, x_shift, axis=1)  # left

        if cfg.dy < 0 or cfg.dy > 0:
            cfg.car_y_fractions += cfg.dy
            cfg.dy = 0
        while cfg.car_y_fractions >= 1 or cfg.car_y_fractions <= -1:
            # y_shift = int(old_pose[1] - pose[1])  # Left with non-fraction part
            y_shift += int(cfg.car_y_fractions)
            print('y_shift', y_shift)
            cfg.car_y_fractions = cfg.car_y_fractions - int(cfg.car_y_fractions)
            log_odd_map = np.roll(log_odd_map, y_shift, axis=0)  # down
        if (y_shift < 0):  # y_shift=-1, we need to reset the last row
            log_odd_map[y_shift:, :] = 0  # cfg.prop_noInfo
        elif (y_shift > 0):  # y_shift=1, we need te reset the first row
            log_odd_map[0:y_shift, :] = 0  # cfg.prop_noInfo
        if (x_shift < 0):  # x_shift=-1, we need to reset the last column
            log_odd_map[:, x_shift:] = 0  # cfg.prop_noInfo
        elif (x_shift > 0):  # x_shift=1, we need to reset the first column
            log_odd_map[:, 0:x_shift] = 0  # cfg.prop_noInfo

        # log_odd_map = np.roll(log_odd_map, x_shift, axis=1)  # down
        log_odd_map = np.roll(log_odd_map, self.Circle_shift[1], axis=0)  # right
        log_odd_map = np.roll(log_odd_map, self.Circle_shift[0], axis=1)  # down
        if (self.Circle_shift[1] < 0):  # y_shift=-1, we need to reset the last row
            log_odd_map[self.Circle_shift[1]:, :] = 0  # cfg.prop_noInfo
        elif (self.Circle_shift[1] > 0):  # y_shift=1, we need te reset the first row
            log_odd_map[0:self.Circle_shift[1], :] = 0  # cfg.prop_noInfo
        if (self.Circle_shift[0] < 0):  # x_shift=-1, we need to reset the last column
            log_odd_map[:, self.Circle_shift[0]:] = 0  # cfg.prop_noInfo
        elif (self.Circle_shift[0] > 0):  # x_shift=1, we need to reset the first column
            log_odd_map[:, 0:self.Circle_shift[0]] = 0  # cfg.prop_noInfo

        scan = scan / cfg.reslution  # Normalize to pixels space
        scan_copy = scan.copy()

        # Get all angles and distances
        angles = np.zeros(len(scan))
        distances = np.zeros(len(scan))

        if x_shift >= 1 or x_shift <= -1 or y_shift >= 1 or y_shift <= -1 \
                or cfg.car_x_fractions > 0 or cfg.car_x_fractions < 0 or cfg.car_y_fractions > 0 or cfg.car_y_fractions < 0:
            if (x_shift > 0 or x_shift < 0 or cfg.car_x_fractions > 0 or cfg.car_x_fractions < 0):
                scan_rotated = self.rotate_points((0, 0), scan, pose[2])
                scan_rotated = np.asarray(scan_rotated).T
                scan_copy[:, 0] = Local_Pose[0] + scan_rotated[:, 0]  # +self.Circle_shift[0]  # Shift by car position
                scan_copy[:, 1] = Local_Pose[1] + scan_rotated[:, 1]  # +self.Circle_shift[1] # Shift by car position
            elif (y_shift > 0 or y_shift < 0 or cfg.car_y_fractions > 0 or cfg.car_y_fractions < 0):
                # if the change in Y is negative, we should + scan_rotated in X and Y
                scan_rotated = self.rotate_points((0, 0), scan, pose[2])
                scan_rotated = np.asarray(scan_rotated).T
                scan_copy[:, 0] = Local_Pose[0] + scan_rotated[:, 0]  # +self.Circle_shift[0] # Shift by car position
                scan_copy[:, 1] = Local_Pose[1] + scan_rotated[:, 1]  # +self.Circle_shift[1] # Shift by car position

            # Visualize & Save Scans
            # temp_map= Actuall_Prob.copy()
            # for p in scan_copy:
            #     if (p[0] > 0 and p[1] > 0 and p[0] < log_odd_map.shape[0] and p[1] < log_odd_map.shape[1]):
            #         temp_map[(int)(p[1]), (int)(p[0])] = 1
            # plt.imshow(temp_map, 'Greys')
            # plt.savefig('D:/Valeo/OGM/OGM_VOC_Scans/' + str(scan_idx) + '.png', dpi=700)
            # plt.show(block=False)
            cloud = np.vstack([scan_copy[:, :2], np.array([Local_Pose[0], Local_Pose[1]])])
            if mode == 'ConvexHull':
                start_time = time.time()
                C_hull = ConvexHull(cloud)  # Add LiDAR centre to convex
                hull_vertices = cloud[C_hull.vertices, :2]
                if not isinstance(hull_vertices, Delaunay):
                    hull_del = Delaunay(hull_vertices)
                    u = hull_del.find_simplex(map_coordinate)
                    affected_map_coordinates = map_coordinate[u >= 0]
                else:
                    raise ValueError("Convex Hull Problem")
                print("Convex Hull Time", str(time.time() - start_time))

            # Visualize Scans
            # temp_map = Actuall_Prob.copy()
            # for p in affected_map_coordinates:
            #     if (p[0] > 0 and p[1] > 0 and p[0] < map.shape[0] and p[1] < map.shape[1]):
            #         temp_map[(int)(p[1]), (int)(p[0])] = 0.3
            # plt.imshow(temp_map, 'Greys')
            # plt.savefig('D:/Valeo/OGM/OGM_VOC_Scans/' + str(scan_idx) + '.png', dpi=700)
            if mode == 'Polygon':
                tupVerts = tuple(cloud)
                min_scan_x = int(np.min(scan_copy[:, 0]))  # +init_pose[0]#+pose[0]
                max_scan_x = int(np.max(scan_copy[:, 0]))  # +init_pose[0]#+pose[0]
                min_scan_y = int(np.min(scan_copy[:, 1]))  # +init_pose[1]#+pose[1]
                max_scan_y = int(np.max(scan_copy[:, 1]))  # +init_pose[1]#+pose[1]
                start_time = time.time()
                x, y = np.meshgrid(np.arange(min_scan_x, max_scan_x),
                                   np.arange(min_scan_y, max_scan_y))  # make a canvas with coordinates
                x, y = x.flatten(), y.flatten()
                points = []
                points = np.vstack((x, y)).T
                p = Path(tupVerts)  # make a polygon
                grid = p.contains_points(points)
                shaped_mask = grid.reshape(points.shape[0], 1)
                affected_map_coordinates = []
                for i in range(shaped_mask.shape[0]):
                    if shaped_mask[i] == True:
                        affected_map_coordinates.append(np.vstack((points[i][0], points[i][1])).T)
                affected_map_coordinates = np.array(affected_map_coordinates).squeeze(axis=1)
                print("Polygon Time", str(time.time() - start_time))
            cfg.retrieve_time += time.time() - start_time
            start_time = time.time()
            for i in range(len(scan_copy)):
                distances[i] = sqrt(
                    (scan_copy[i][0] - Local_Pose[0]) ** 2 + (scan_copy[i][1] - Local_Pose[1]) ** 2)  # r
                angles[i] = atan2(scan_copy[i][1] - Local_Pose[1], scan_copy[i][0] - Local_Pose[0])  # b
                while (angles[i] > np.pi):
                    angles[i] -= 2. * np.pi
                while (angles[i] < -np.pi):
                    angles[i] += 2. * np.pi

            new_probs = np.ones(len(affected_map_coordinates)) * 0.5
            angle_diff = np.zeros(len(affected_map_coordinates))

            guranteed_range = 50 / cfg.reslution
            for i in range(len(affected_map_coordinates)):
                dist_car = sqrt((affected_map_coordinates[i][0] - Local_Pose[0]) ** 2 +
                                (affected_map_coordinates[i][1] - Local_Pose[1]) ** 2)  # * cfg.reslution
                angle = atan2(affected_map_coordinates[i][1] - Local_Pose[1],
                              affected_map_coordinates[i][0] - Local_Pose[0])
                while (angle > np.pi):
                    angle -= 2. * np.pi
                while (angle < -np.pi):
                    angle += 2. * np.pi
                angle_difference = np.min(np.abs(angles - angle))
                angle_diff[i] = angle_difference
                nearest_scanpoint_idx = (np.abs(angles - angle)).argmin()
                if (np.abs(angle_difference) < cfg.beam_azimuth / 2):
                    if (dist_car < (distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                        new_probs[i] = cfg.prop_free
                    elif (dist_car <= (distances[nearest_scanpoint_idx] + cfg.wall_depth / 2.)):
                        new_probs[i] = cfg.prop_occ
                    else:
                        new_probs[i] = cfg.prop_noInfo
                else:
                    if ((dist_car <= guranteed_range) & (
                            dist_car < distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                        new_probs[i] = cfg.prop_free
                    elif ((dist_car > guranteed_range) & (
                            dist_car < distances[nearest_scanpoint_idx] - cfg.wall_depth / 2.)):
                        new_probs[
                            i] = cfg.prop_free  # Oxford_linear_decay(init_pose[0]+guranteed_range, init_pose[0]+(cfg.range/cfg.reslution), init_pose[0]+(dist_car))
                    else:
                        new_probs[i] = cfg.prop_noInfo

            for i in range(len(affected_map_coordinates)):
                log_odd_map[affected_map_coordinates[i][1], affected_map_coordinates[i][0]] = \
                    log_odd_map[affected_map_coordinates[i][1], affected_map_coordinates[i][0]] + \
                    np.log(new_probs[i]) - np.log((1 - new_probs[i]))

            cfg.affected_points += len(affected_map_coordinates)
            cfg.update_time += time.time() - start_time
            cfg.count_scans += 1
            print("Convex Hull points = ", len(affected_map_coordinates), "Convex Hull Time update",
                  str(time.time() - start_time))

        return log_odd_map, x_shift, y_shift

    def Oxford_linear_decay(self, min_distance, max_distance, distance):
        angle = 0
        for idx in range(0, self.cfg.object_reflectivity):
            angle += 4.5 / (int(idx / 10) + 1)
        # Linear decay between the distance and the max range
        value = self.cfg.prop_free + ((distance - min_distance) * (self.cfg.prop_noInfo - self.cfg.prop_free) / (
                max_distance - min_distance))
        if (angle < 45):
            if (value > self.cfg.prop_free):
                value -= (45 - angle) * (self.cfg.prop_noInfo - self.cfg.prop_free) / 90
            if (value < self.cfg.prop_free):  # to make sure that the value didn't exceed the prop_free
                value = self.cfg.prop_free
        if (angle > 45):
            if (value < self.cfg.prop_noInfo):
                value += (angle - 45) * (self.cfg.prop_noInfo - self.cfg.prop_free) / 90
            if (value > self.cfg.prop_noInfo):  # to make sure that the value didn't exceed the prop_free
                value = self.cfg.prop_noInfo
        return value

    # ------------------------------------------------------------------------------
    # Main update function
    # ------------------------------------------------------------------------------
    # - Update region = 'ConvexHull' or 'Polygon'
    # - Update equation:
    #   1-Update_ogm (The bayesian rule)
    # 	2-Update_ogm_Log_odd (The log odd format) much faster
    def draw_ogm_map(self, x_abs, y_abs, yaw_abs, speed, fullscan, full_scans_timestamps=None,
                   method='ConvexHull', use_log_odds=True):
        # Poses, fullscans, full_scans_timestamps, speed_filtered = read_oxford_data("2014-05-06-12-54-54")
        # Poses, fullscans, full_scans_timestamps, speed_filtered = read_scala_data("20150326_112329_Grandeur2")#read_data(scala or oxford) or Carla
        # x_abs,y_abs,yaw_abs,xy_timestamps = Poses

        pose = [x_abs / self.cfg.reslution,  # x_abs[loop_idx]/self.cfg.reslution
                y_abs / self.cfg.reslution, yaw_abs]
        line_length = self.cfg.circle_r + (2 * speed)

        if (self.old_pose == 0):
            self.Initial_pose[2] = yaw_abs
            self.old_pose = self.Initial_pose.copy()
            self.cfg.p_local[0] = self.Initial_pose[0] + line_length * np.cos(np.pi + yaw_abs)
            self.cfg.p_local[1] = self.Initial_pose[1] + line_length * np.sin(np.pi + yaw_abs)

        self.cfg.dx += pose[0] - self.old_pose[0]
        self.cfg.dy += pose[1] - self.old_pose[1]
        previous_local = self.cfg.p_local.copy()
        if ((self.cfg.dx >= 0 and (yaw_abs >= 0 and yaw_abs < 90) or (yaw_abs > 270 and yaw_abs < 360)) or # forward
            (self.cfg.dx <= 0 and (yaw_abs >= 90 and yaw_abs < 270))):  # forward #80 instead of 90 because of odometry
            self.cfg.p_local[0] = self.Initial_pose[0] + line_length * np.cos(np.pi + yaw_abs)
            self.cfg.p_local[1] = self.Initial_pose[1] + line_length * (np.sin(np.pi + yaw_abs))
        else:  # backward
            self.cfg.p_local[0] = self.Initial_pose[0] + line_length * np.cos(yaw_abs)
            self.cfg.p_local[1] = self.Initial_pose[1] + line_length * (np.sin(yaw_abs))

        change_in_circle_x = self.cfg.p_local[0] - previous_local[0]
        change_in_circle_y = self.cfg.p_local[1] - previous_local[1]
        print('change_in_circle_x', change_in_circle_x)
        print('change_in_circle_y', change_in_circle_y)
        self.Circle_shift = [change_in_circle_x, change_in_circle_y]

        if use_log_odds:
            self.log_odd_map, x_shift, y_shift = self.update_ogm_Log_odd(method, fullscan, pose, self.cfg.p_local,
                                                                         self.cfg, self.map_coordinates,
                                                                         self.log_odd_map)
        else:
            self.map, x_shift, y_shift = self.update_ogm(method, fullscan, pose, self.cfg.p_local, self.cfg,
                                                         self.map_coordinates, self.map)
        self.map = 1.0 - 1. / (1 + np.exp(self.log_odd_map))

        self.old_pose = pose.copy()
