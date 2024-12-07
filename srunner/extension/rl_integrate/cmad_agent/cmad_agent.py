from __future__ import annotations, print_function

import json
import weakref

import carla

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.extension.rl_integrate.data.sensor_data_provider import SensorDataProvider
from srunner.extension.rl_integrate.data.simulator import Simulator
from srunner.scenariomanager.timer import GameTime


def sensor_name_to_bp(camera_type: str):
    """Convert sensor name to blueprint"""
    camera_type = camera_type.lower()
    if "rgb" in camera_type:
        return "sensor.camera.rgb"
    elif "depth" in camera_type:
        return "sensor.camera.depth"
    elif "semseg" in camera_type:
        return "sensor.camera.semantic_segmentation"
    elif "lidar" in camera_type:
        return "sensor.lidar.ray_cast"
    elif "bev" in camera_type:
        # Use a pseudo sensor type to make it compatible with AgentWrapper
        return "sensor.other.gnss"
    else:
        raise ValueError("Unknown sensor name: {}".format(camera_type))


class CmadAgent(AutonomousAgent):
    """
    All agent used in macad-gym should inherit from this class.

    You can override setup(), run_step() and destroy() methods

    Note:
        call super().setup(actor_config) in your setup() method to parse sensors from actor_config
    """

    _camera_transforms = [
        {
            "x": 1.8,
            "y": 0.0,
            "z": 1.7,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
        },  # for normal car
        {
            "x": 3.6,
            "y": 0.0,
            "z": 2.5,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
        },  # for firetruck
        {
            "x": -5.5,
            "y": 0.0,
            "z": 2.8,
            "pitch": -15.0,
            "yaw": 0.0,
            "roll": 0.0,
        },  # back camera
    ]

    def setup(self, config):
        if isinstance(config, dict):
            self.actor_config = config
        else:
            with open(config, "r") as f:
                self.actor_config = json.load(f)

        self.camera_obs = None
        self.sensor_list = []

        weak_self = weakref.ref(self)
        self.callbacks = [
            Simulator.add_callback(
                lambda snapshot: CmadAgent.on_carla_tick(weak_self, snapshot)
            )
        ]
        self.parse_sensors()

    def sensors(self):
        return self.sensor_list

    def parse_sensors(self):
        """Parse sensors from actor config to make it compatible for setup_sensors in AgentWrapper"""
        camera_types = self.actor_config.get("camera_type", [])

        if not isinstance(camera_types, list):
            camera_types = [camera_types]

        for camera_type in camera_types:
            if camera_type == "":
                continue

            sensor_spec = {
                "id": camera_type,
                "type": sensor_name_to_bp(camera_type),
                "width": self.actor_config["x_res"],
                "height": self.actor_config["y_res"],
                "attachment_type": carla.AttachmentType.Rigid,
            }

            # Use default values to meet AgentWrapper's requirement
            if sensor_spec["type"].startswith("sensor.camera"):
                sensor_spec.update({"fov": 90})
            elif sensor_spec["type"].startswith("sensor.lidar"):
                sensor_spec.update(
                    {
                        "range": 10.0,
                        "rotation_frequency": 10.0,
                        "channels": 32,
                        "upper_fov": 10.0,
                        "lower_fov": -30.0,
                        "points_per_second": 56000,
                    }
                )

            camera_pos = self.actor_config.get("camera_position", 0)
            if isinstance(camera_pos, dict):
                sensor_spec.update(camera_pos)
            else:
                sensor_spec.update(self._camera_transforms[camera_pos])

            self.sensor_list.append(sensor_spec)

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        timestamp = GameTime.get_time()
        control = self.run_step(self.camera_obs, timestamp)
        return control

    def destroy(self):
        for callback in self.callbacks:
            Simulator.remove_callback(callback)

        self.camera_obs = None
        self.actor_config = None
        self.sensor_list = []
        self.callbacks = []

    @staticmethod
    def on_carla_tick(weak_self, snapshot: carla.WorldSnapshot):
        """Update obs on carla tick"""
        self: CmadAgent = weak_self()
        if (not self) or (not len(self.sensor_list) > 0):
            return

        frame = (
            next(iter(self.camera_obs.values()))[0]
            if (self.camera_obs is not None) and len(self.camera_obs) > 0
            else -1
        )
        while frame < snapshot.frame:
            try:
                self.camera_obs = self.sensor_interface.get_data(0.1)
                frame = next(iter(self.camera_obs.values()))[0]
            except:
                pass

            if not len(self.sensor_list) > 0:
                break

        try:
            actor_id = self.actor_config.get("actor_id", "global")
            SensorDataProvider.update_camera_data(actor_id, self.camera_obs)
        except:
            pass
