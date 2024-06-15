import numpy as np
import sys
import math
class Drone:
    def __init__(self, id, starting, destination, waypoints, vx, vy, vz, ax, ay, az):

        self.id = id
        self.starting = starting
        self.destination = destination
        self.waypoints = waypoints
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def pos_new(self):

        self.vx = vx
        self.vy = vy
        self.vz = vz

