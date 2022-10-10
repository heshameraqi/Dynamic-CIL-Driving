# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import collections
import math

import numpy as np

from . import city_track


def compare(x, y):
    return collections.Counter(x) == collections.Counter(y)



# Constants Used for the high level commands


REACH_GOAL = 0.0
GO_STRAIGHT = 5.0
TURN_RIGHT = 4.0
TURN_LEFT = 3.0
LANE_FOLLOW = 2.0


# Auxiliary algebra function
def angle_between(v1, v2):
    return np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))


def sldist(c1, c2): return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


def signal(v1, v2):
    return np.cross(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)


class Planner(object):

    def __init__(self, city_name):

        self._city_track = city_track.CityTrack(city_name)

        self._commands = []

        self.route = None
        self.commands = None
        self.far_from_intersection = None
        self.previous_command = None  # Safer to get an error instead of LANE_FOLLOW
        self._previous_node = None  # Used to reset the route planner (get_next_command function)
        self._next_node = None
        self.OGM_route_reroute = False

    # The original CIL paper planner used in our ITSC 20 paper
    """def get_next_command(self, source, source_ori, target, target_ori):
        '''
        Computes the full plan and returns the next command,
        Args
            source: source position
            source_ori: source orientation
            target: target position
            target_ori: target orientation
        Returns
            a command ( Straight,Lane Follow, Left or Right)
        '''

        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        # reach the goal

        if self._city_track.is_at_goal(track_source, track_target):
            return REACH_GOAL

        if (self._city_track.is_at_new_node(track_source)
                and self._city_track.is_away_from_intersection(track_source)):

            route = self._city_track.compute_route(track_source, source_ori,
                                                   track_target, target_ori)
            if route is None:
                raise RuntimeError('Impossible to find route')

            self._commands = self._route_to_commands(route)

            self.far_from_intersection = self._city_track.is_far_away_from_route_intersection(track_source)
            if self.far_from_intersection:
                return LANE_FOLLOW
            else:
                if self._commands:
                    return self._commands[0]
                else:
                    return LANE_FOLLOW
        else:
            self.far_from_intersection = self._city_track.is_far_away_from_route_intersection(track_source)
            if self.far_from_intersection:
                return LANE_FOLLOW

            # If there is computed commands
            if self._commands:
                return self._commands[0]
            else:
                return LANE_FOLLOW"""

    # Our new simpler planner
    def get_next_command(self, source, source_ori, target, target_ori):
        '''
        Computes the full plan and returns the next command,
        Args
            source: source position
            source_ori: source orientation
            target: target position
            target_ori: target orientation
        Returns
            a command ( Straight,Lane Follow, Left or Right)
        '''

        # From CARLA position (GPS) to planner nodes grid map
        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        # reach the goal
        if self._city_track.is_at_goal(track_source, track_target):
            return REACH_GOAL

        # Re-execute A* if needed and add a wall behind car before
        exited_route = False
        if self._previous_node is not None and track_source != self._previous_node:
            if track_source == self.route[self._next_node]:
                self._previous_node = track_source
                self._next_node += 1
            elif track_source == self.route[self._next_node+1]:
                self._previous_node = track_source
                self._next_node += 2
            else:
                # if self._city_track.is_away_from_intersection(track_source):  # because putting a wall behind the car might becomes more challenging in terms of which direction should be blocked
                exited_route = True
        if self.OGM_route_reroute or exited_route or self._previous_node is None:
            if self.OGM_route_reroute:
                self.OGM_route_reroute = False
            self.route = self._city_track.compute_route(track_source, source_ori, track_target, target_ori)
            self._previous_node = track_source
            self._next_node = 1  # the node after the car position
            self.commands = [LANE_FOLLOW for i in range(len(self.route))]
            for i in range(0, len(self.route)):
                if self.route[i] in self._city_track.get_intersection_nodes():
                    current = self.route[i]
                    past = self.route[i - 1]
                    future = self.route[i + 1]
                    past_to_current = np.array([current[0] - past[0], current[1] - past[1]])
                    current_to_future = np.array([future[0] - current[0], future[1] - current[1]])
                    angle = signal(current_to_future, past_to_current)
                    if angle < -0.1: command = TURN_RIGHT
                    elif angle > 0.1: command = TURN_LEFT
                    else: command = GO_STRAIGHT

                    s = max(0, i-3)  # put -3 for 3 before intersection, making it lower causes the intersections to be known by car late and the network drives the car to miss it
                    e = min(len(self.route), i+2)  # put +2 for 1 after intersection
                    self.commands[s:e] = [command] * (e-s)

        for i in range(0, len(self.route)):
            if self.route[i] == track_source:
                self.far_from_intersection = self._city_track.is_far_away_from_route_intersection(track_source)  # For visualization
                self.previous_command = self.commands[i]
                return self.commands[i]
        return self.previous_command

    # Our new simpler planner (slow but stable)
    """def get_next_command(self, source, source_ori, target, target_ori):
        '''
        Computes the full plan and returns the next command,
        Args
            source: source position
            source_ori: source orientation
            target: target position
            target_ori: target orientation
        Returns
            a command ( Straight,Lane Follow, Left or Right)
        '''

        # From CARLA position (GPS) to planner nodes grid map
        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        # reach the goal

        if self._city_track.is_at_goal(track_source, track_target):
            return REACH_GOAL

        # is_away_from_intersection measures if node - intersection > 1: because putting a wall behind the car might becomes more challenging in terms of which direction should be blocked
        if self.OGM_route_reroute or (
                track_source != self._previous_node and self._city_track.is_away_from_intersection(
                track_source)) or self._previous_node is None:
            if self.OGM_route_reroute:
                self.OGM_route_reroute = False
            self._previous_node = track_source
            self.route = self._city_track.compute_route(track_source, source_ori, track_target, target_ori)
            self.commands = [LANE_FOLLOW for i in range(len(self.route))]
            for i in range(0, len(self.route)):
                if self.route[i] in self._city_track.get_intersection_nodes():
                    current = self.route[i]
                    past = self.route[i - 1]
                    future = self.route[i + 1]
                    past_to_current = np.array([current[0] - past[0], current[1] - past[1]])
                    current_to_future = np.array([future[0] - current[0], future[1] - current[1]])
                    angle = signal(current_to_future, past_to_current)
                    if angle < -0.1:
                        command = TURN_RIGHT
                    elif angle > 0.1:
                        command = TURN_LEFT
                    else:
                        command = GO_STRAIGHT

                    s = max(0,
                            i - 3)  # put -3 for 3 before intersection, making it lower causes the intersections to be known by car late and the network drives the car to miss it
                    e = min(len(self.route), i + 2)  # put +2 for 1 after intersection
                    self.commands[s:e] = [command] * (e - s)

        for i in range(0, len(self.route)):
            if self.route[i] == track_source:
                self.far_from_intersection = self._city_track.is_far_away_from_route_intersection(
                    track_source)  # For visualization
                self.previous_command = self.commands[i]
                return self.commands[i]
        return self.previous_command"""

    def get_shortest_path_distance(
            self,
            source,
            source_ori,
            target,
            target_ori):

        distance = 0
        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        current_pos = track_source

        route = self._city_track.compute_route(track_source, source_ori,
                                               track_target, target_ori)
        # No Route, distance is zero
        if route is None:
            return 0.0

        for node_iter in route:
            distance += sldist(node_iter, current_pos)
            current_pos = node_iter

        # We multiply by these values to convert distance to world coordinates
        return distance * self._city_track.get_pixel_density() \
               * self._city_track.get_node_density()

    def is_there_posible_route(self, source, source_ori, target, target_ori):

        track_source = self._city_track.project_node(source)
        track_target = self._city_track.project_node(target)

        return not self._city_track.compute_route(
            track_source, source_ori, track_target, target_ori) is None

    def test_position(self, source):

        node_source = self._city_track.project_node(source)

        return self._city_track.is_away_from_intersection(node_source)

    def _route_to_commands(self, route):

        """
        from the shortest path graph, transform it into a list of commands

        :param route: the sub graph containing the shortest path
        :return: list of commands encoded from 0-5
        """

        commands_list = []

        for i in range(0, len(route)):
            if route[i] not in self._city_track.get_intersection_nodes():
                continue

            current = route[i]
            past = route[i - 1]
            future = route[i + 1]

            past_to_current = np.array(
                [current[0] - past[0], current[1] - past[1]])
            current_to_future = np.array(
                [future[0] - current[0], future[1] - current[1]])
            angle = signal(current_to_future, past_to_current)

            if angle < -0.1:
                command = TURN_RIGHT
            elif angle > 0.1:
                command = TURN_LEFT
            else:
                command = GO_STRAIGHT

            commands_list.append(command)

        return commands_list
