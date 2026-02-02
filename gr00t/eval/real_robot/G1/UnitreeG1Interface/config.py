"""
G1 Hardware Configuration

Parses YAML config file and provides structured access to hardware parameters.
"""

import os
import yaml
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


# Default config path (relative to this file)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "g1_hardware.yaml"


class HardwareConfig:
    """
    G1 Robot Hardware Configuration.
    
    Loads config from YAML and provides access to:
    - Motor indices and groups
    - Default positions
    - PD gains
    - Control frequency
    - Camera settings
    - Hand configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Load hardware configuration from YAML file.
        
        Args:
            config_path: Path to g1_hardware.yaml. If None, uses default.
        """
        if config_path is None:
            config_path = str(DEFAULT_CONFIG_PATH)
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Cache computed values
        self._kp_gains: Optional[np.ndarray] = None
        self._kd_gains: Optional[np.ndarray] = None
        self._default_positions: Optional[np.ndarray] = None
    
    # =========================================================================
    # Motor Indices
    # =========================================================================
    
    @property
    def leg_indices(self) -> List[int]:
        """Leg motor indices (0-11)."""
        return self._config["joints"]["legs"]["indices"]
    
    @property
    def left_leg_indices(self) -> List[int]:
        """Left leg motor indices (0-5)."""
        return self._config["joints"]["legs"]["left"]
    
    @property
    def right_leg_indices(self) -> List[int]:
        """Right leg motor indices (6-11)."""
        return self._config["joints"]["legs"]["right"]
    
    @property
    def waist_indices(self) -> List[int]:
        """Waist motor indices (12-14)."""
        return self._config["joints"]["waist"]["indices"]
    
    @property
    def waist_locked(self) -> bool:
        """Whether waist is locked (23dof model)."""
        return self._config["joints"]["waist"].get("locked", False)
    
    @property
    def arm_indices(self) -> List[int]:
        """Arm motor indices (15-28)."""
        return self._config["joints"]["arms"]["indices"]
    
    @property
    def left_arm_indices(self) -> List[int]:
        """Left arm motor indices (15-21)."""
        return self._config["joints"]["arms"]["left"]
    
    @property
    def right_arm_indices(self) -> List[int]:
        """Right arm motor indices (22-28)."""
        return self._config["joints"]["arms"]["right"]
    
    @property
    def lower_body_indices(self) -> List[int]:
        """Lower body indices = legs (0-11)."""
        return self.leg_indices
    
    @property
    def upper_body_indices(self) -> List[int]:
        """Upper body indices = waist + arms (12-28)."""
        return self.waist_indices + self.arm_indices
    
    @property
    def num_motors(self) -> int:
        """Total number of motors."""
        return self._config["joints"]["num_motors"]
    
    # =========================================================================
    # Control Parameters
    # =========================================================================
    
    @property
    def control_frequency(self) -> float:
        """Control frequency in Hz."""
        return self._config["control"]["frequency"]
    
    @property
    def control_dt(self) -> float:
        """Control timestep in seconds."""
        return self._config["control"]["dt"]
    
    @property
    def mode_pr(self) -> int:
        """Ankle control mode (0: PR, 1: AB)."""
        return self._config["control"]["mode_pr"]
    
    @property
    def default_positions(self) -> np.ndarray:
        """Default positions for all 29 motors."""
        if self._default_positions is None:
            pos = self._config["joints"]["default_position"]
            self._default_positions = np.array(
                pos["legs"] + pos["waist"] + pos["arms"], 
                dtype=np.float32
            )
        return self._default_positions
    
    @property
    def kp_gains(self) -> np.ndarray:
        """Kp gains for all 29 motors."""
        if self._kp_gains is None:
            joints = self._config["joints"]
            self._kp_gains = np.array(
                joints["legs"]["kp"] + 
                joints["waist"]["kp"] + 
                joints["arms"]["kp"],
                dtype=np.float32
            )
        return self._kp_gains
    
    @property
    def kd_gains(self) -> np.ndarray:
        """Kd gains for all 29 motors."""
        if self._kd_gains is None:
            joints = self._config["joints"]
            self._kd_gains = np.array(
                joints["legs"]["kd"] + 
                joints["waist"]["kd"] + 
                joints["arms"]["kd"],
                dtype=np.float32
            )
        return self._kd_gains
    
    # =========================================================================
    # Network
    # =========================================================================
    
    @property
    def dds_interface(self) -> str:
        """DDS network interface (e.g., 'eno2')."""
        return self._config["network"]["interface"]
    
    @property
    def robot_ip(self) -> str:
        """Robot IP address."""
        return self._config["network"]["robot_ip"]
    
    # =========================================================================
    # Hands
    # =========================================================================
    
    @property
    def hands_enabled(self) -> bool:
        """Whether hands are enabled."""
        return self._config["hands"]["enabled"]
    
    @property
    def hands_type(self) -> str:
        """Hand type (inspire, dex3-1)."""
        return self._config["hands"]["type"]
    
    @property
    def left_hand_num_joints(self) -> int:
        """Number of joints in left hand."""
        return self._config["hands"]["left"]["num_joints"]
    
    @property
    def right_hand_num_joints(self) -> int:
        """Number of joints in right hand."""
        return self._config["hands"]["right"]["num_joints"]
    
    # =========================================================================
    # Cameras
    # =========================================================================
    
    @property
    def cameras_enabled(self) -> bool:
        """Whether cameras are enabled."""
        return self._config["cameras"]["enabled"]
    
    @property
    def camera_names(self) -> List[str]:
        """List of configured camera names."""
        cameras = self._config["cameras"]
        return [k for k in cameras.keys() if k != "enabled" and isinstance(cameras[k], dict)]
    
    def get_camera_config(self, camera_name: str) -> Dict[str, Any]:
        """Get configuration for a specific camera."""
        return self._config["cameras"][camera_name]
    
    # =========================================================================
    # Raw Access
    # =========================================================================
    
    @property
    def raw_config(self) -> Dict[str, Any]:
        """Access raw config dictionary."""
        return self._config
    
    # =========================================================================
    # Joint Indices by Name
    # =========================================================================
    
    @property
    def joint_indices(self) -> Dict[str, int]:
        """Get joint name to index mapping."""
        return self._config.get("joint_indices", {})
    
    def get_joint_index(self, joint_name: str) -> int:
        """Get index for a specific joint by name."""
        return self._config["joint_indices"][joint_name]
    
    # =========================================================================
    # Joystick
    # =========================================================================
    
    @property
    def joystick_enabled(self) -> bool:
        """Whether joystick is enabled."""
        return self._config.get("joystick", {}).get("enabled", False)
    
    @property
    def joystick_button_indices(self) -> Dict[str, int]:
        """Joystick button indices mapping."""
        return self._config.get("joystick", {}).get("button_indices", {})
    
    @property
    def joystick_axis_indices(self) -> Dict[str, int]:
        """Joystick axis byte offsets."""
        return self._config.get("joystick", {}).get("axis_indices", {})