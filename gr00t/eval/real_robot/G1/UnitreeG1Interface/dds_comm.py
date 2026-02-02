"""
G1 DDS Communication

Handles low-level DDS pub/sub for robot state and commands.
Based on Unitree SDK2 Python official examples.
"""

from typing import Optional, List
import numpy as np
import threading

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


# G1 constants
G1_NUM_MOTOR = 29
CONTROL_MODE_PR = 0  # Pitch/Roll mode
CONTROL_MODE_AB = 1  # A/B parallel mode


class G1StateReceiver:
    """Receives robot state via DDS subscriber."""

    def __init__(self):
        self._low_state: Optional[LowState_] = None
        self._lock = threading.Lock()
        self._subscriber: Optional[ChannelSubscriber] = None
        self._mode_machine: int = 0
        self._initialized = False

    def init(self):
        """Initialize DDS subscriber."""
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._state_callback, 10)
        self._initialized = True

    def _state_callback(self, msg: LowState_):
        """Callback for state updates."""
        with self._lock:
            self._low_state = msg
            self._mode_machine = msg.mode_machine

    def wait_for_state(self, timeout: float = 5.0) -> bool:
        """Wait until first state is received."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self._low_state is not None:
                return True
            time.sleep(0.01)
        return False

    @property
    def mode_machine(self) -> int:
        """Get current mode_machine value."""
        return self._mode_machine

    def get_motor_positions(self) -> np.ndarray:
        """Get current motor positions (29,)."""
        with self._lock:
            if self._low_state is None:
                return np.zeros(G1_NUM_MOTOR)
            return np.array([
                self._low_state.motor_state[i].q 
                for i in range(G1_NUM_MOTOR)
            ])

    def get_motor_velocities(self) -> np.ndarray:
        """Get current motor velocities (29,)."""
        with self._lock:
            if self._low_state is None:
                return np.zeros(G1_NUM_MOTOR)
            return np.array([
                self._low_state.motor_state[i].dq 
                for i in range(G1_NUM_MOTOR)
            ])

    def get_motor_torques(self) -> np.ndarray:
        """Get current motor torques (29,)."""
        with self._lock:
            if self._low_state is None:
                return np.zeros(G1_NUM_MOTOR)
            return np.array([
                self._low_state.motor_state[i].tau_est 
                for i in range(G1_NUM_MOTOR)
            ])

    def get_imu_data(self) -> dict:
        """Get IMU data (quaternion, rpy, gyro, accel)."""
        with self._lock:
            if self._low_state is None:
                return {
                    "quaternion": np.zeros(4),
                    "rpy": np.zeros(3),
                    "gyro": np.zeros(3),
                    "accel": np.zeros(3),
                }
            imu = self._low_state.imu_state
            return {
                "quaternion": np.array(imu.quaternion),
                "rpy": np.array(imu.rpy),
                "gyro": np.array(imu.gyroscope),
                "accel": np.array(imu.accelerometer),
            }

    def get_wireless_remote(self) -> bytes:
        """Get raw wireless remote data for joystick parsing."""
        with self._lock:
            if self._low_state is None:
                return bytes(40)  # Return empty bytes
            return self._low_state.wireless_remote

    @property
    def low_state(self):
        """Access raw low_state (for advanced use)."""
        return self._low_state


class G1CommandSender:
    """Sends motor commands via DDS publisher."""

    def __init__(self):
        self._publisher: Optional[ChannelPublisher] = None
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._initialized = False

    def init(self):
        """Initialize DDS publisher."""
        self._publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._publisher.Init()
        self._initialized = True

    def send_motor_commands(
        self,
        positions: np.ndarray,
        mode_machine: int,
        kp: np.ndarray,
        kd: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        torques: Optional[np.ndarray] = None,
        mode_pr: int = CONTROL_MODE_PR,
    ):
        """
        Send motor commands with PD control.
        
        Args:
            positions: Target positions (29,)
            mode_machine: Current mode_machine from state
            kp: Position gains (29,)
            kd: Velocity gains (29,)
            velocities: Target velocities (29,), default zeros
            torques: Feedforward torques (29,), default zeros
            mode_pr: Control mode (PR or AB)
        """
        if velocities is None:
            velocities = np.zeros(G1_NUM_MOTOR)
        if torques is None:
            torques = np.zeros(G1_NUM_MOTOR)

        self._low_cmd.mode_pr = mode_pr
        self._low_cmd.mode_machine = mode_machine

        for i in range(G1_NUM_MOTOR):
            self._low_cmd.motor_cmd[i].mode = 1  # Enable
            self._low_cmd.motor_cmd[i].q = float(positions[i])
            self._low_cmd.motor_cmd[i].dq = float(velocities[i])
            self._low_cmd.motor_cmd[i].tau = float(torques[i])
            self._low_cmd.motor_cmd[i].kp = float(kp[i])
            self._low_cmd.motor_cmd[i].kd = float(kd[i])

        # Calculate CRC and send
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._publisher.Write(self._low_cmd)


def init_dds(interface: str = ""):
    """
    Initialize DDS channel factory.
    
    Args:
        interface: Network interface (e.g., "eth0", "eno2"). 
                   Empty string for default.
    """
    if interface:
        ChannelFactoryInitialize(0, interface)
    else:
        ChannelFactoryInitialize(0)
