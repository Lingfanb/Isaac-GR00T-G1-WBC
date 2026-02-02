"""
Dex3-1 Hand Control for Unitree G1

Based on official Unitree SDK example: g1_move_hands_example.py
Uses separate DDS topics for left and right hand control.
"""

from typing import Optional
import numpy as np
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_


# Dex3-1 has 7 motors per hand
MOTOR_NUM_HAND = 7

# Default PD gains
HAND_KP = [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
HAND_KD = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


def make_hand_mode(motor_index: int) -> int:
    """
    Build the mode bitfield for hand motor command.
    
    Args:
        motor_index: Motor index (0-6)
        
    Returns:
        Mode value with status and timeout flags
    """
    status = 0x01
    timeout = 0x01
    mode = (motor_index & 0x0F)
    mode |= (status << 4)   # bits [4..6]
    mode |= (timeout << 7)  # bit 7
    return mode


class Dex3HandInterface:
    """
    Interface for Dex3-1 dexterous hands.
    
    Provides high-level API for hand control:
        - Get current hand joint positions
        - Send hand joint commands
    
    Usage:
        hands = Dex3HandInterface()
        hands.init()
        
        left_pos = hands.get_left_positions()
        hands.send_left_command(target_positions)
        
    DDS Topics:
        - rt/dex3/left/cmd, rt/dex3/left/state
        - rt/dex3/right/cmd, rt/dex3/right/state
    """

    def __init__(self):
        # Publishers
        self._left_cmd_pub: Optional[ChannelPublisher] = None
        self._right_cmd_pub: Optional[ChannelPublisher] = None
        
        # Subscribers
        self._left_state_sub: Optional[ChannelSubscriber] = None
        self._right_state_sub: Optional[ChannelSubscriber] = None
        
        # State buffers
        self._left_state: Optional[HandState_] = None
        self._right_state: Optional[HandState_] = None
        self._lock = threading.Lock()
        
        # PD gains
        self.kp = np.array(HAND_KP)
        self.kd = np.array(HAND_KD)
        
        self._initialized = False

    def init(self):
        """Initialize DDS publishers and subscribers for hand control."""
        # Publishers
        self._left_cmd_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self._right_cmd_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._left_cmd_pub.Init()
        self._right_cmd_pub.Init()
        
        # Subscribers
        self._left_state_sub = ChannelSubscriber("rt/dex3/left/state", HandState_)
        self._right_state_sub = ChannelSubscriber("rt/dex3/right/state", HandState_)
        self._left_state_sub.Init(self._left_state_callback, 10)
        self._right_state_sub.Init(self._right_state_callback, 10)
        
        self._initialized = True

    def _left_state_callback(self, msg: HandState_):
        """Callback for left hand state."""
        with self._lock:
            self._left_state = msg

    def _right_state_callback(self, msg: HandState_):
        """Callback for right hand state."""
        with self._lock:
            self._right_state = msg

    def wait_for_state(self, timeout: float = 3.0) -> bool:
        """Wait until first state is received from both hands."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self._left_state is not None and self._right_state is not None:
                return True
            time.sleep(0.01)
        return False

    # =========================================================================
    # Get State
    # =========================================================================

    def get_left_positions(self) -> np.ndarray:
        """Get left hand joint positions (7,)."""
        with self._lock:
            if self._left_state is None:
                return np.zeros(MOTOR_NUM_HAND)
            return np.array([
                self._left_state.motor_state[i].q 
                for i in range(MOTOR_NUM_HAND)
            ])

    def get_right_positions(self) -> np.ndarray:
        """Get right hand joint positions (7,)."""
        with self._lock:
            if self._right_state is None:
                return np.zeros(MOTOR_NUM_HAND)
            return np.array([
                self._right_state.motor_state[i].q 
                for i in range(MOTOR_NUM_HAND)
            ])

    def get_positions(self) -> tuple:
        """Get both hand positions."""
        return self.get_left_positions(), self.get_right_positions()

    # =========================================================================
    # Send Commands
    # =========================================================================

    def send_left_command(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        torques: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ):
        """
        Send command to left hand.
        
        Args:
            positions: Target joint positions (7,)
            velocities: Target velocities (7,), default zeros
            torques: Feedforward torques (7,), default zeros
            kp: Position gains (7,), default HAND_KP
            kd: Velocity gains (7,), default HAND_KD
        """
        cmd = self._build_hand_cmd(positions, velocities, torques, kp, kd)
        self._left_cmd_pub.Write(cmd)

    def send_right_command(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        torques: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ):
        """
        Send command to right hand.
        
        Args:
            positions: Target joint positions (7,)
            velocities: Target velocities (7,), default zeros
            torques: Feedforward torques (7,), default zeros
            kp: Position gains (7,), default HAND_KP
            kd: Velocity gains (7,), default HAND_KD
        """
        cmd = self._build_hand_cmd(positions, velocities, torques, kp, kd)
        self._right_cmd_pub.Write(cmd)

    def send_both_commands(
        self,
        left_positions: np.ndarray,
        right_positions: np.ndarray,
    ):
        """Send commands to both hands."""
        self.send_left_command(left_positions)
        self.send_right_command(right_positions)

    def _build_hand_cmd(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        torques: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> HandCmd_:
        """Build hand command message."""
        if velocities is None:
            velocities = np.zeros(MOTOR_NUM_HAND)
        if torques is None:
            torques = np.zeros(MOTOR_NUM_HAND)
        if kp is None:
            kp = self.kp
        if kd is None:
            kd = self.kd
        
        cmd = unitree_hg_msg_dds__HandCmd_()
        
        for i in range(MOTOR_NUM_HAND):
            cmd.motor_cmd[i].mode = make_hand_mode(i)
            cmd.motor_cmd[i].q = float(positions[i])
            cmd.motor_cmd[i].dq = float(velocities[i])
            cmd.motor_cmd[i].tau = float(torques[i])
            cmd.motor_cmd[i].kp = float(kp[i])
            cmd.motor_cmd[i].kd = float(kd[i])
        
        return cmd

    # =========================================================================
    # Utility
    # =========================================================================

    def open_hands(self):
        """Open both hands (all fingers extended)."""
        open_pos = np.zeros(MOTOR_NUM_HAND)
        self.send_both_commands(open_pos, open_pos)

    def close_hands(self):
        """Close both hands (grasp position)."""
        # Approximate grasp position
        close_pos = np.array([0.8, 0.5, 0.5, -1.0, -1.0, -1.0, -1.0])
        close_pos_right = np.array([0.8, -0.5, -0.5, 1.0, 1.0, 1.0, 1.0])
        self.send_both_commands(close_pos, close_pos_right)
