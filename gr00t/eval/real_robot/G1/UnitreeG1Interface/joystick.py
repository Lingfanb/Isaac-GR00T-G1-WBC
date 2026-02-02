"""
Unitree Remote Controller Interface

Parses joystick/button data from Unitree wireless controller.
Based on official SDK example: wireless_controller.py
"""

from typing import Optional
import struct
import threading


class UnitreeRemoteController:
    """
    Unitree wireless remote controller parser.
    
    Reads joystick and button data from robot's low state.
    """

    def __init__(self):
        # Joystick axes (-1.0 to 1.0)
        self.Lx: float = 0.0  # Left stick X
        self.Ly: float = 0.0  # Left stick Y
        self.Rx: float = 0.0  # Right stick X
        self.Ry: float = 0.0  # Right stick Y

        # Buttons (0 or 1)
        self.L1: int = 0
        self.L2: int = 0
        self.R1: int = 0
        self.R2: int = 0
        self.A: int = 0
        self.B: int = 0
        self.X: int = 0
        self.Y: int = 0
        self.Up: int = 0
        self.Down: int = 0
        self.Left: int = 0
        self.Right: int = 0
        self.Select: int = 0
        self.Start: int = 0
        self.F1: int = 0
        self.F3: int = 0
        
        self._lock = threading.Lock()

    def parse(self, remote_data: bytes):
        """
        Parse raw remote controller data.
        
        Args:
            remote_data: Raw bytes from low_state.wireless_remote
        """
        with self._lock:
            self._parse_axes(remote_data)
            self._parse_buttons(remote_data[2], remote_data[3])

    def _parse_axes(self, data: bytes):
        """Parse joystick axes."""
        self.Lx = struct.unpack('<f', data[4:8])[0]
        self.Rx = struct.unpack('<f', data[8:12])[0]
        self.Ry = struct.unpack('<f', data[12:16])[0]
        self.Ly = struct.unpack('<f', data[20:24])[0]

    def _parse_buttons(self, data1: int, data2: int):
        """Parse button states."""
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def is_start_pressed(self) -> bool:
        """Check if Start button is pressed."""
        return self.Start == 1

    def is_select_pressed(self) -> bool:
        """Check if Select button is pressed."""
        return self.Select == 1

    def is_emergency_stop(self) -> bool:
        """Check if L1 (emergency stop) is pressed."""
        return self.L1 == 1

    def get_locomotion_command(self) -> tuple:
        """
        Get locomotion velocity command from joysticks.
        
        Returns:
            (vx, vy, vyaw): Forward velocity, lateral velocity, yaw rate
        """
        with self._lock:
            vx = self.Ly      # Forward/back
            vy = self.Lx      # Left/right
            vyaw = self.Rx    # Turn
        return vx, vy, vyaw

    def get_state_dict(self) -> dict:
        """Get all controller state as a dictionary."""
        with self._lock:
            return {
                "axes": {
                    "Lx": self.Lx,
                    "Ly": self.Ly,
                    "Rx": self.Rx,
                    "Ry": self.Ry,
                },
                "buttons": {
                    "L1": self.L1,
                    "L2": self.L2,
                    "R1": self.R1,
                    "R2": self.R2,
                    "A": self.A,
                    "B": self.B,
                    "X": self.X,
                    "Y": self.Y,
                    "Up": self.Up,
                    "Down": self.Down,
                    "Left": self.Left,
                    "Right": self.Right,
                    "Select": self.Select,
                    "Start": self.Start,
                    "F1": self.F1,
                    "F3": self.F3,
                },
            }
