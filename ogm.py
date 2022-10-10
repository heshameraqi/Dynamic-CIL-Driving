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
import math


# ------------------------------------------------------------------------------
# Configurations class
# ------------------------------------------------------------------------------
class configuration:
    def __init__(self):
        self.width = 80  # meters, 100 is a good value, actually meters/10 due to CARLA LiDAR pointcloud metrics
        self.height = 80  # meters, 100 is a good value, (OGM assumed square when normalizing scan points)
        self.resolution = 0.5  # meters per grid cell, 0.25 is a good value. width and height should be multiple of it
        self.wall_depth = 1.5 / self.resolution  # [meters] #alpha
        self.beam_azimuth = 2.0 * np.pi / 180.0  # [radian]

        self.voc_circle_default_radius = 10 / self.resolution  # in cells
        self.speed_scale = 11  # For VoC

        self.range = 80 / self.resolution  # [meters] #300 for oxford, 50 for Scala and CARLA AUC dataset

        # self.prop_free = 0.45
        # self.prop_occ = 0.55
        self.log_prop_occ = 0.9
        self.log_prop_free = -0.2

        self.dx = 0.0
        self.dy = 0.0
        self.retrieve_time = 0
        self.update_time = 0
        self.affected_points = 0
        self.count_scans = 0
        self.object_reflectivity = 10
        self.car_x_fractions = 0
        self.car_y_fractions = 0
        self.pose_local = np.array([0, 0, 0])
        self.current_velocity = 0


class OGM:
    def __init__(self):
        # Load configurations
        self.cfg = configuration()
        # Create one-dimensional arrays for all x and y of map
        y = range((int)(self.cfg.height / self.cfg.resolution))
        x = range((int)(self.cfg.width / self.cfg.resolution))
        self.map_coordinates = np.array(list(itertools.product(x, y)))

        self.map_positions = np.array([np.tile(np.arange(0, (int)(self.cfg.height / self.cfg.resolution))[:, None],
                                               (1, (int)(self.cfg.width / self.cfg.resolution))),
                                       np.tile(np.arange(0, (int)(self.cfg.width / self.cfg.resolution))[:, None].T,
                                               ((int)(self.cfg.height / self.cfg.resolution), 1))])

        self.previous_pose_world = -1  # unset
        self.previous_pose_local = -1  # unset
        self.pose_local = -1  # unset
        self.voc_radius = -1  # unset
        self.map_fullscan = -1  # unset (fullscan pointcloud in map space not relative to vehicle)
        self.map_centre_cell = np.array([self.cfg.width/(2*self.cfg.resolution),
                                         self.cfg.height/(2*self.cfg.resolution), 0])
        self.map = None  # can be log odds format
        self.map_without_dynamic = None  # can be log odds format
        self.map_without_dynamic_prob = None  # what class user usually needs

    # ------------------------------------------------------------------------------
    # OGM Local Functions
    # ------------------------------------------------------------------------------
    def update_ogm_bf_log_odds(self, mode, cloud, local_pose):
        # Transform scanpoints from local car coordinates to local map coordinates (scale and rotate)
        cloud = cloud / self.cfg.resolution  # Normalize to pixels space
        cell_angle = local_pose[2] - np.pi / 2
        c, s = np.cos(cell_angle), np.sin(cell_angle)
        R = np.array(((c, -s), (s, c)))
        cloud[:, 0:2] = np.dot(cloud[:, 0:2], R)
        cloud[:, 0] += local_pose[0]
        cloud[:, 1] += local_pose[1]
        # self.map_fullscan = cloud  # For visualization
        cloud = cloud[:, 0:2]  # remove z
        # original_cloud = cloud.copy()  # For visualization

        # Get distances and angles for each point & shift cloud by wall_depth away along beams direction
        angles = np.zeros(len(cloud))  # All angles
        distances = np.ones(len(cloud))  # All distances
        for i in range(len(cloud)):  # start_time = time.time()
            angles[i] = atan2(cloud[i, 1] - local_pose[1], cloud[i, 0] - local_pose[0])
            """while angles[i] > np.pi:
                angles[i] -= 2. * np.pi"""
            while angles[i] < -np.pi:
                angles[i] += 2. * np.pi
            cloud[i, 0] += self.cfg.wall_depth * np.cos(angles[i])
            cloud[i, 1] += self.cfg.wall_depth * np.sin(angles[i])
            distances[i] = sqrt((cloud[i, 0] - local_pose[0]) ** 2 + (cloud[i, 1] - local_pose[1]) ** 2)

        # Get affected area
        cloud = np.vstack([np.array([local_pose[0:2]]), cloud[:, :2]])  # Add sensor location point
        if mode == 'ConvexHull':
            # start_time = time.time()
            C_hull = ConvexHull(cloud)  # Add LiDAR centre to convex
            hull_vertices = cloud[C_hull.vertices, :2]
            if not isinstance(hull_vertices, Delaunay):
                hull_del = Delaunay(hull_vertices)
                u = hull_del.find_simplex(self.map_coordinates)
                affected_map_coordinates = self.map_coordinates[u >= 0]
            else:
                raise ValueError("Convex Hull Problem")
            # print("Convex Hull Time", str(time.time() - start_time))
        elif mode == 'Polygon':  # TODO: needed revisiting
            tupVerts = tuple(cloud)
            min_scan_x = int(np.min(cloud[:, 0]))  # +init_pose[0]#+pose[0]
            max_scan_x = int(np.max(cloud[:, 0]))  # +init_pose[0]#+pose[0]
            min_scan_y = int(np.min(cloud[:, 1]))  # +init_pose[1]#+pose[1]
            max_scan_y = int(np.max(cloud[:, 1]))  # +init_pose[1]#+pose[1]
            # start_time = time.time()
            x, y = np.meshgrid(np.arange(min_scan_x, max_scan_x),
                               np.arange(min_scan_y, max_scan_y))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            p = Path(tupVerts)  # make a polygon
            grid = p.contains_points(points)
            shaped_mask = grid.reshape(points.shape[0], 1)
            affected_map_coordinates = []
            for i in range(shaped_mask.shape[0]):
                if shaped_mask[i]:
                    affected_map_coordinates.append(np.vstack((points[i][0], points[i][1])).T)
            affected_map_coordinates = np.array(affected_map_coordinates).squeeze(axis=1)
        cloud = cloud[1:, :]  # Remove sensor location point from scanpoints

        # Visualize fullscan and affected area
        visualize = False
        if visualize:
            plt.figure()
            temp_map = self.map.copy()
            for p in affected_map_coordinates:
                if 0 < p[0] < self.map.shape[0] and 0 < p[1] < self.map.shape[1]:
                    temp_map[int(p[1]), int(p[0])] = 0.3
            for p in original_cloud:
                if 0 < p[0] < self.map.shape[0] and 0 < p[1] < self.map.shape[1]:
                    temp_map[int(p[1]), int(p[0])] = 0.8
            for p in cloud:  # After shifting
                if 0 < p[0] < self.map.shape[0] and 0 < p[1] < self.map.shape[1]:
                    temp_map[int(p[1]), int(p[0])] = 0.6
            arrow_length = 4
            plt.arrow(local_pose[0] - arrow_length * np.cos(local_pose[2]),
                      local_pose[1] + arrow_length * np.sin(local_pose[2]),
                      arrow_length * np.cos(local_pose[2]), -arrow_length * np.sin(local_pose[2]),
                      length_includes_head= True,
                      head_width=arrow_length, head_length=0.5*arrow_length, width=0.5*arrow_length, color="r")
            plt.imshow(temp_map)
            plt.show()
            # plt.savefig('/media/heraqi/data/heraqi/int-end-to-end-ad/ogm/' + str(scan_idx) + '.png', dpi=700)

        for i in range(len(affected_map_coordinates)):
            cell_dist = sqrt((affected_map_coordinates[i][0] - local_pose[0]) ** 2 +
                            (affected_map_coordinates[i][1] - local_pose[1]) ** 2)  # * self.cfg.resolution
            cell_angle = atan2(affected_map_coordinates[i][1] - local_pose[1],
                          affected_map_coordinates[i][0] - local_pose[0])
            """while angle > np.pi:
                angle -= 2. * np.pi"""
            while cell_angle < -np.pi:
                cell_angle += 2. * np.pi

            idx_beams = np.abs(angles - cell_angle) < (self.cfg.beam_azimuth / 2)
            if idx_beams.any():
                scan_dist = np.min(distances[idx_beams])
                if cell_dist < (scan_dist - self.cfg.wall_depth):
                    new_prob = self.cfg.log_prop_free
                elif cell_dist <= scan_dist:
                    new_prob = self.cfg.log_prop_occ
                else:  # Cell after wall depth
                    continue  # no update
            else:
                nearest_scanpoint_idx = (np.abs(angles - cell_angle)).argmin()
                if cell_dist < distances[nearest_scanpoint_idx]:
                    new_prob = self.cfg.log_prop_free
                else:
                    continue  # no update
            # guaranteed_range = 50 / self.cfg.resolution for Oxford LiDAR dataset
            '''else:  # For Oxford LiDAR dataset because it has different modes of operation in different ranges
                if ((dist_car <= guranteed_range) & (dist_car < distances[nearest_scanpoint_idx] - self.cfg.wall_depth / 2.)):
                    new_prob = self.cfg.prop_free
                elif ((dist_car > guranteed_range) & (dist_car < distances[nearest_scanpoint_idx] - self.cfg.wall_depth / 2.)):
                    new_prob = self.cfg.prop_free  # Oxford_linear_decay(init_pose[0]+guranteed_range, init_pose[0]+(self.cfg.range/self.cfg.resolution), init_pose[0]+(dist_car))
                else:
                    new_prob = self.cfg.prop_noInfo'''
            self.map[affected_map_coordinates[i][1], affected_map_coordinates[i][0]] += new_prob

        # self.cfg.update_time += time.time() - start_time
        # print("Convex Hull points = ", len(affected_map_coordinates),
        #       "Convex Hull Time update", str(time.time() - start_time))
        self.cfg.affected_points += len(affected_map_coordinates)
        self. cfg.count_scans += 1

    def add_dynamic_scans(self, dynamic_objects_scans):
        # Transform scanpoints from local car coordinates to local map coordinates (scale and rotate)
        cloud = dynamic_objects_scans / self.cfg.resolution  # Normalize to pixels space
        cell_angle = self.pose_local[2] - np.pi / 2
        c, s = np.cos(cell_angle), np.sin(cell_angle)
        R = np.array(((c, -s), (s, c)))
        cloud[:, 0:2] = np.dot(cloud[:, 0:2], R)
        cloud[:, 0] += self.pose_local[0]
        cloud[:, 1] += self.pose_local[1]
        # self.map_fullscan = cloud  # For visualization
        cloud = cloud[:, 0:2]  # remove z
        # original_cloud = cloud.copy()  # For visualization

        for p in cloud:
            if 0 < p[0] < self.map.shape[0] and 0 < p[1] < self.map.shape[1]:
                self.map[int(p[1]), int(p[0])] += 1

    # ------------------------------------------------------------------------------
    # Main update function
    # ------------------------------------------------------------------------------
    # Scan points and sensor position are in meters, update region = 'ConvexHull' or 'Polygon'
    def draw_ogm_map(self, x_abs, y_abs, yaw_abs, speed, fullscan, dynamic_objects_scans, method='ConvexHull'):
        pose_world = [x_abs/self.cfg.resolution, -y_abs/self.cfg.resolution, -yaw_abs]

        # Process yaw
        while pose_world[2] < 0:
            pose_world[2] += 360
        pose_local = [0, 0, np.deg2rad(pose_world[2])]  # Yaw is converted to radians

        # Calculate position circle
        self.voc_radius = self.cfg.voc_circle_default_radius + (self.cfg.speed_scale * speed) # TODO: if vehicle goes backward remove the pi
        self.voc_radius = min(self.voc_radius, self.cfg.width/(2*self.cfg.resolution)-10)  # Prevent circle from getting out of map in very high speeds
        pose_circle = [self.map_centre_cell[0] - self.voc_radius * np.cos(pose_local[2]),
                       self.map_centre_cell[1] + self.voc_radius * np.sin(pose_local[2])]

        # For first full scan
        if self.previous_pose_world == -1:
            # Create Empty Map, zeros log_odds mean occupancy probability of 0.5
            self.map = np.zeros((int(self.cfg.height / self.cfg.resolution),
                                 int(self.cfg.width / self.cfg.resolution)))

            # Set previous local and world poses
            self.previous_pose_world = pose_world
            self.previous_pose_local = pose_circle.copy()

        # Remove dynamic objects added then Shift Map
        if self.map_without_dynamic is not None:
            self.map = self.map_without_dynamic.copy()
        shift = [pose_world[0] - self.previous_pose_world[0] + self.previous_pose_local[0] - pose_circle[0],
                 -pose_world[1] + self.previous_pose_world[1] + self.previous_pose_local[1] - pose_circle[1]]
        left_shift = int(math.modf(shift[0])[1])
        down_shift = -int(math.modf(shift[1])[1])
        if left_shift > 0:  # shift left
            self.map = np.pad(self.map, ((0, 0), (0, left_shift)), mode='constant')[:, left_shift:]
        elif left_shift < 0:  # shift right
            self.map = np.pad(self.map, ((0, 0), (-left_shift, 0)), mode='constant')[:, :left_shift]
        if down_shift > 0:  # shift down
            self.map = np.pad(self.map, ((down_shift, 0), (0, 0)), mode='constant')[:-down_shift, :]
        elif down_shift < 0:  # shift up
            self.map = np.pad(self.map, ((0, -down_shift), (0, 0)), mode='constant')[-down_shift:, :]

        # Shift car with fractional shifts
        pose_local[0] = pose_circle[0] + math.modf(shift[0])[0]
        pose_local[1] = pose_circle[1] + math.modf(shift[1])[0]
        self.pose_local = pose_local  # for class users to know car location, like visualization

        # Update map with Bayes Filter (comment for faster experimentation)
        self.update_ogm_bf_log_odds(method, fullscan, pose_local)

        # Copy map and then manually add dynamic object obstacles
        self.map_without_dynamic = self.map.copy()
        self.add_dynamic_scans(dynamic_objects_scans)

        # Convert to probability
        exp = np.exp(self.map_without_dynamic)
        self.map_without_dynamic_prob = np.divide(exp, 1 + exp)

        # TODO: apply morphological opening as in [Robust Free Space Detection in Occupancy Grid Maps by Methods of
        #  Image Analysis and Dynamic B-Spline Contour Tracking.]

        # Set previous world and local poses
        self.previous_pose_world = pose_world.copy()
        self.previous_pose_local = pose_local.copy()
