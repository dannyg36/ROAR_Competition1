"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.throttle = Throttle()
    
    async def initialize(self) -> None:
        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )


    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
         # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 10) % len(self.maneuverable_waypoints)]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -10 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Proportional controller to control the vehicle's speed towards 40 m/s
        throttle_control, brake_control = self.throttle.compute_throttle_and_brake(vehicle_velocity_norm, self.current_waypoint_idx, waypoint_to_follow, self.maneuverable_waypoints, vehicle_location, vehicle_rotation, delta_time=0.1)
        print(f"Throttle: {throttle_control}, Brake: {brake_control}, Speed: {vehicle_velocity_norm}")

        control = {
            "throttle": throttle_control,
            "steer": steer_control,
            "brake": brake_control,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control

class Throttle:
    def __init__(self, target_speed: float = 40.0, kp: float = 0.1, ki: float = 0.01, kd: float = 0.05) -> None:
        self.base_target_speed = target_speed
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute_throttle_and_brake(self, current_speed: float, current_waypoint_idx: int, waypoint_to_follow: roar_py_interface.RoarPyWaypoint, maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint], vehicle_location: np.ndarray, vehicle_rotation: np.ndarray, delta_time: float) -> Tuple[float, float]:
        speed_error = self.base_target_speed - current_speed
        self.integral += speed_error * delta_time
        derivative = (speed_error - self.prev_error) / delta_time
        self.prev_error = speed_error

        if current_speed > 25:
            lookahead_idx = (current_waypoint_idx + 100) % len(maneuverable_waypoints)
            waypoint_to_follow = maneuverable_waypoints[lookahead_idx]

        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
        average_curve = np.abs(delta_heading)

        throttle = self.kp * speed_error + self.ki * self.integral + self.kd * derivative
        throttle = np.clip(throttle, 0.0, 1.0)
        if average_curve > 0.05 and current_speed > 25:
            brake = 0.8
            throttle = 0.5
        elif average_curve > 0.05:
            brake = 0.25
            throttle = 0.6
        else:
            brake = 0
        return throttle, brake