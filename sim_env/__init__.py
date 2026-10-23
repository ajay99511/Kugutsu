"""MuJoCo scenes (assets/*.xml) and simulation wrappers."""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class RobotEnv:
    def __init__(self):
        xml_path = Path(__file__).parent / "assets" / "pick_cube.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def step(self, action: np.ndarray) -> bool:
        """action: [target_x, target_y, target_z, gripper_close_amount]"""
        if not self.viewer.is_running():
            return False

        # Update Mocap XYZ target
        self.data.mocap_pos[0] = action[:3]

        # Update finger actuators (0.0 = open, 0.025 = closed/pinched)
        finger_pos = action[3] * 0.025
        self.data.ctrl[0] = finger_pos
        self.data.ctrl[1] = finger_pos

        # Step physics 10 times per camera frame
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.viewer.sync()
        return True

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()
