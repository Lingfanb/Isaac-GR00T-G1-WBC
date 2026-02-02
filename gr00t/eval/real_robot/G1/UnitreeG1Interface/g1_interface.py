"""
Unitree G1 Robot Interface

High-level API for G1 robot control.
Combines DDS communication, configuration, joystick, and optional hand control.
"""

from typing import Any, Dict, Optional, List
import numpy as np
import time

from .config import HardwareConfig
from .dds_comm import G1StateReceiver, G1CommandSender, init_dds, G1_NUM_MOTOR
from .joystick import UnitreeRemoteController


class G1RobotInterface:
    """
    High-level interface for Unitree G1 robot.
    
    Combines state receiver, command sender, joystick, and optional hand control.
    Provides a clean API similar to LeRobot for consistency with SO100.
    
    Usage:
        robot = G1RobotInterface()
        robot.connect()
        
        obs = robot.get_observation()
        robot.send_action({"positions": target_positions})
        
        # Check joystick
        if robot.joystick.is_start_pressed():
            print("Start pressed!")
        
        robot.disconnect()
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize G1 interface.
        
        Args:
            config_path: Path to g1_hardware.yaml. If None, uses default.
        """
        self.config = HardwareConfig(config_path)
        self._state_receiver: Optional[G1StateReceiver] = None
        self._command_sender: Optional[G1CommandSender] = None
        self._connected = False
        
        # Joystick/Remote controller
        self.joystick = UnitreeRemoteController()
        
        # Camera placeholder (to be implemented)
        self._cameras: Dict[str, Any] = {}

    def connect(self):
        """Connect to robot hardware."""
        if self._connected:
            return
        
        # Initialize DDS
        init_dds(self.config.dds_interface)
        
        # Initialize state receiver
        self._state_receiver = G1StateReceiver()
        self._state_receiver.init()
        
        # Initialize command sender
        self._command_sender = G1CommandSender()
        self._command_sender.init()
        
        # Wait for first state
        if not self._state_receiver.wait_for_state(timeout=3.0):
            raise RuntimeError("Failed to receive robot state within timeout")
        
        self._connected = True
        print("G1 robot connected successfully")

    def disconnect(self):
        """Disconnect from robot hardware."""
        self._connected = False
        self._state_receiver = None
        self._command_sender = None
        print("G1 robot disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connected

    # =========================================================================
    # Observation
    # =========================================================================

    def get_observation(self) -> Dict[str, Any]:
        """
        Get current robot observation.
        
        Returns:
            Dict containing:
                - motor_positions: (29,) array
                - motor_velocities: (29,) array
                - motor_torques: (29,) array
                - imu: dict with quaternion, rpy, gyro, accel
                - arm_positions: (14,) array (both arms)
                - left_arm_positions: (7,) array
                - right_arm_positions: (7,) array
                - joystick: dict with axes and buttons
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")
        
        positions = self._state_receiver.get_motor_positions()
        velocities = self._state_receiver.get_motor_velocities()
        torques = self._state_receiver.get_motor_torques()
        imu = self._state_receiver.get_imu_data()
        
        # Update joystick state from low_state
        self._update_joystick()
        
        # Extract arm positions
        left_arm_idx = self.config.left_arm_indices
        right_arm_idx = self.config.right_arm_indices
        
        obs = {
            # Full body
            "motor_positions": positions,
            "motor_velocities": velocities,
            "motor_torques": torques,
            "imu": imu,
            
            # Arms (for manipulation)
            "arm_positions": positions[self.config.arm_indices],
            "left_arm_positions": positions[left_arm_idx],
            "right_arm_positions": positions[right_arm_idx],
            
            # Legs (for locomotion)
            "leg_positions": positions[self.config.leg_indices],
            
            # Joystick
            "joystick": self.joystick.get_state_dict(),
        }
        
        return obs

    def _update_joystick(self):
        """Update joystick state from low_state."""
        if self._state_receiver._low_state is not None:
            remote_data = self._state_receiver._low_state.wireless_remote
            self.joystick.parse(remote_data)

    def get_arm_positions(self) -> np.ndarray:
        """Get arm motor positions (14,) for both arms."""
        positions = self._state_receiver.get_motor_positions()
        return positions[self.config.arm_indices]

    def get_left_arm_positions(self) -> np.ndarray:
        """Get left arm motor positions (7,)."""
        positions = self._state_receiver.get_motor_positions()
        return positions[self.config.left_arm_indices]

    def get_right_arm_positions(self) -> np.ndarray:
        """Get right arm motor positions (7,)."""
        positions = self._state_receiver.get_motor_positions()
        return positions[self.config.right_arm_indices]

    # =========================================================================
    # Action
    # =========================================================================

    def send_action(self, action: Dict[str, Any]):
        """
        Send action to robot.
        
        Args:
            action: Dict containing one of:
                - positions: (29,) target positions for all motors
                - left_arm: (7,) left arm positions
                - right_arm: (7,) right arm positions
                - arm_positions: (14,) both arm positions
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")
        
        # Get current positions as base
        current_positions = self._state_receiver.get_motor_positions()
        target_positions = current_positions.copy()
        
        # Apply action based on what's provided
        if "positions" in action:
            target_positions = np.array(action["positions"])
        else:
            if "left_arm" in action:
                for i, idx in enumerate(self.config.left_arm_indices):
                    target_positions[idx] = action["left_arm"][i]
            if "right_arm" in action:
                for i, idx in enumerate(self.config.right_arm_indices):
                    target_positions[idx] = action["right_arm"][i]
            if "arm_positions" in action:
                for i, idx in enumerate(self.config.arm_indices):
                    target_positions[idx] = action["arm_positions"][i]
        
        # Send command
        self._command_sender.send_motor_commands(
            positions=target_positions,
            mode_machine=self._state_receiver.mode_machine,
            kp=self.config.kp_gains,
            kd=self.config.kd_gains,
            mode_pr=self.config.mode_pr,
        )

    def send_arm_action(self, left_arm: np.ndarray, right_arm: np.ndarray):
        """
        Send arm commands only (keep other joints unchanged).
        
        Args:
            left_arm: (7,) left arm target positions
            right_arm: (7,) right arm target positions
        """
        self.send_action({
            "left_arm": left_arm,
            "right_arm": right_arm,
        })

    def send_full_body_action(self, positions: np.ndarray):
        """
        Send full body action.
        
        Args:
            positions: (29,) target positions for all motors
        """
        self.send_action({"positions": positions})

    # =========================================================================
    # Joystick Convenience Methods
    # =========================================================================

    def is_start_pressed(self) -> bool:
        """Check if Start button is pressed on joystick."""
        self._update_joystick()
        return self.joystick.is_start_pressed()

    def is_stop_pressed(self) -> bool:
        """Check if Select (stop) button is pressed."""
        self._update_joystick()
        return self.joystick.is_select_pressed()

    def is_emergency_stop(self) -> bool:
        """Check if L1 (emergency stop) is pressed."""
        self._update_joystick()
        return self.joystick.is_emergency_stop()

    def get_locomotion_command(self) -> tuple:
        """Get locomotion velocity command from joysticks (vx, vy, vyaw)."""
        self._update_joystick()
        return self.joystick.get_locomotion_command()

    # =========================================================================
    # Utility
    # =========================================================================

    def go_to_default_position(self, duration: float = 3.0):
        """
        Smoothly move to default standing position.
        
        Args:
            duration: Time in seconds to reach default position
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")
        
        start_positions = self._state_receiver.get_motor_positions()
        target_positions = self.config.default_positions
        
        dt = self.config.control_dt
        steps = int(duration / dt)
        
        for step in range(steps):
            ratio = step / steps
            interpolated = (1.0 - ratio) * start_positions + ratio * target_positions
            
            self._command_sender.send_motor_commands(
                positions=interpolated,
                mode_machine=self._state_receiver.mode_machine,
                kp=self.config.kp_gains,
                kd=self.config.kd_gains,
            )
            time.sleep(dt)
        
        print("Reached default position")
