from __future__ import annotations

import carla
import numpy as np

from srunner.extension.rl_integrate.cmad_agent.action.action_interface import (
    AbstractAction,
    ActionInterface,
)


class DirectAction(AbstractAction):
    def __init__(self, action, duration=1):
        super().__init__(action, duration)

    def run_step(self, actor: carla.Walker):
        """Return a carla control signal based on the actor type and the action

        Args:
            actor (carla.Actor): The actor to run the action

        Returns:
            carla.WalkerControl: The control signal
        """
        self.duration -= 1

        direction = carla.Vector3D()
        yaw = actor.get_transform().rotation.yaw

        direction_str = self.action["direction"]
        x, y = DirectAction.action_transform(
            yaw, DirectAction.get_direction_vector(direction_str)
        )
        direction.x, direction.y = float(x), float(y)

        actor_control = carla.WalkerControl()
        speed = self.action["speed"]
        actor_control.speed = speed
        actor_control.direction = direction
        return actor_control

    @staticmethod
    def action_transform(yaw, direction):
        yaw = np.radians(-yaw)
        rotation_matrix = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        )
        inverse_rotation_matrix = np.transpose(rotation_matrix)
        return np.dot(inverse_rotation_matrix, direction)

    @staticmethod
    def get_direction_vector(direction_str):
        """Convert direction string to carla.Vector3D"""

        direction_map = {
            "stay": (0, 0),
            "front": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        x, y = direction_map[direction_str]
        return np.array([x, y])


class WalkerAction(ActionInterface):
    def __init__(self, action_config: dict):
        """Initialize the action converter for low-level action space

        Args:
            action_config (dict): A dictionary of action config
        """
        super().__init__(action_config)

    def convert_single_action(
        self, action: "tuple[list, float]", done_state: bool = False
    ):
        """Convert the action of a model output to an AbstractAction instance

        Args:
            action: Action for a single actor
            done_state (bool): Whether the actor is done. If done, return a stop action

        Returns:
            DirectAction: A direct action instance
        """
        if done_state:
            return self.stop_action(env_action=False)
        else:
            return DirectAction({"direction": action[0], "speed": action[1]})

    def get_action_mask(self, actor, action_space):
        """Low-level action is always applicable"""
        return True

    def stop_action(self, env_action: bool = True, use_discrete: bool = False):
        """Return the stop action representation in low-level action space

        Args:
            env_action (bool): Whether using env action space
            use_discrete (bool): Whether using discrete action space

        Returns:
            DirectAction: if env_action is False, return the stop action in the action space of the action handler.
            EnvAction: a valid action in the env action space
        """

        if not env_action:
            return DirectAction({"direction": "stay", "speed": 0.0})

        return [0, 0]
