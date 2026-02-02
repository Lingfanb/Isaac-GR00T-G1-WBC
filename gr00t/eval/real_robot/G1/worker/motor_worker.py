"""
Motor Worker for Unitree G1 Robot

Runs in a separate process to handle:
- Robot state reading at high frequency
- Command execution from shared memory
- Joystick monitoring for start/stop/emergency

Uses multiprocessing shared memory with dictionary-style access.
"""

import time
import numpy as np
from multiprocessing import Process, Value, Manager
from multiprocessing.managers import DictProxy
from dataclasses import dataclass
from typing import Optional, Dict, Any
import copy

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from UnitreeG1Interface import G1RobotInterface, HardwareConfig


NUM_MOTORS = 29
MOTOR_WORKER_FREQUENCY = 1000.0  # Hz - default control frequency for motor worker


@dataclass
class MotorWorkerConfig:
    """Configuration for motor worker."""
    control_frequency: float = MOTOR_WORKER_FREQUENCY
    enable_command_execution: bool = True
    verbose: bool = False


class SharedState:
    """
    Shared state dictionary between processes.
    
    使用方式:
        state = shared_state.get()
        print(state["left_arm"])  # 左臂位置 (7,)
        print(state["right_leg"])  # 右腿位置 (6,)
    
    可用的 key:
        - "motor_positions": 所有电机位置 (29,)
        - "motor_velocities": 所有电机速度 (29,)
        - "motor_torques": 所有电机力矩 (29,)
        
        身体部位 (位置):
        - "left_leg": 左腿 (6,)
        - "right_leg": 右腿 (6,)
        - "waist": 腰部 (3,)
        - "left_arm": 左臂 (7,)
        - "right_arm": 右臂 (7,)
        
        IMU:
        - "imu_quaternion": 四元数 (4,)
        - "imu_rpy": roll/pitch/yaw (3,)
        - "imu_gyro": 角速度 (3,)
        - "imu_accel": 加速度 (3,)
        
        控制标志:
        - "timestamp": 时间戳
        - "frame_count": 帧计数
    """
    
    # Joint group indices (based on G1 configuration)
    JOINT_GROUPS = {
        "left_leg": list(range(0, 6)),      # 0-5
        "right_leg": list(range(6, 12)),    # 6-11
        "waist": list(range(12, 15)),       # 12-14
        "left_arm": list(range(15, 22)),    # 15-21
        "right_arm": list(range(22, 29)),   # 22-28
    }
    
    def __init__(self, manager: Manager, is_writer: bool = False):
        """
        Initialize shared state.
        
        Args:
            manager: multiprocessing Manager instance
            is_writer: True if this is the writer (motor worker)
        """
        self._manager = manager
        self._is_writer = is_writer
        
        # Create shared dictionary
        if is_writer:
            self._dict = manager.dict()
            self._init_default_values()
        else:
            self._dict = None  # Will be set by set_dict()
    
    def _init_default_values(self):
        """Initialize with default values."""
        self._dict["motor_positions"] = np.zeros(NUM_MOTORS, dtype=np.float32).tolist()
        self._dict["motor_velocities"] = np.zeros(NUM_MOTORS, dtype=np.float32).tolist()
        self._dict["motor_torques"] = np.zeros(NUM_MOTORS, dtype=np.float32).tolist()
        
        # Body parts
        for name, indices in self.JOINT_GROUPS.items():
            self._dict[name] = np.zeros(len(indices), dtype=np.float32).tolist()
        
        # IMU
        self._dict["imu_quaternion"] = [0.0, 0.0, 0.0, 1.0]
        self._dict["imu_rpy"] = [0.0, 0.0, 0.0]
        self._dict["imu_gyro"] = [0.0, 0.0, 0.0]
        self._dict["imu_accel"] = [0.0, 0.0, 0.0]
        
        # Metadata
        self._dict["timestamp"] = 0.0
        self._dict["frame_count"] = 0
    
    def set_dict(self, shared_dict: DictProxy):
        """Set the shared dictionary (for reader process)."""
        self._dict = shared_dict
    
    def get_dict(self) -> DictProxy:
        """Get the underlying shared dictionary."""
        return self._dict
    
    def update_from_observation(self, obs: Dict[str, Any]):
        """
        Update state from robot observation.
        
        Args:
            obs: Observation dict from G1RobotInterface.get_observation()
        """
        if not self._is_writer:
            raise RuntimeError("Only writer can update state")
        
        # Full arrays
        positions = obs["motor_positions"]
        velocities = obs["motor_velocities"]
        torques = obs["motor_torques"]
        
        self._dict["motor_positions"] = positions.tolist()
        self._dict["motor_velocities"] = velocities.tolist()
        self._dict["motor_torques"] = torques.tolist()
        
        # Body parts (extracted from positions)
        for name, indices in self.JOINT_GROUPS.items():
            self._dict[name] = positions[indices].tolist()
        
        # IMU
        imu = obs.get("imu", {})
        self._dict["imu_quaternion"] = list(imu.get("quaternion", [0, 0, 0, 1]))
        self._dict["imu_rpy"] = list(imu.get("rpy", [0, 0, 0]))
        self._dict["imu_gyro"] = list(imu.get("gyro", [0, 0, 0]))
        self._dict["imu_accel"] = list(imu.get("accel", [0, 0, 0]))
        
        # Metadata
        self._dict["timestamp"] = time.time()
        self._dict["frame_count"] = self._dict.get("frame_count", 0) + 1
    
    def get(self, key: Optional[str] = None):
        """
        Get state value(s).
        
        Args:
            key: Specific key to get. If None, returns all as dict.
            
        Returns:
            numpy array or full dict
            
        Usage:
            state.get("left_arm")  # -> np.array([...])
            state.get()  # -> {"left_arm": [...], "right_arm": [...], ...}
        """
        if key is not None:
            value = self._dict[key]
            if isinstance(value, list):
                return np.array(value, dtype=np.float32)
            return value
        
        # Return all as numpy arrays
        result = {}
        for k, v in self._dict.items():
            if isinstance(v, list):
                result[k] = np.array(v, dtype=np.float32)
            else:
                result[k] = v
        return result
    
    def __getitem__(self, key: str):
        """Dictionary-style access: state["left_arm"]"""
        return self.get(key)


class SharedCommand:
    """
    Shared command dictionary between processes.
    
    使用方式 (inference process):
        command.set("left_arm", target_positions)
        command.set("right_arm", target_positions)
        command.mark_valid()
        
    使用方式 (motor worker):
        if command.is_valid():
            targets = command.get_target_positions()
            command.mark_consumed()
            
    可用的 key (用于设置命令):
        - "positions": 设置所有29个电机位置
        - "left_leg": 左腿 (6,)
        - "right_leg": 右腿 (6,)
        - "waist": 腰部 (3,)
        - "left_arm": 左臂 (7,)
        - "right_arm": 右臂 (7,)
    """
    
    JOINT_GROUPS = SharedState.JOINT_GROUPS
    
    def __init__(self, manager: Manager, is_writer: bool = False):
        """
        Initialize shared command.
        
        Args:
            manager: multiprocessing Manager instance
            is_writer: True if this is the writer (inference process)
        """
        self._manager = manager
        self._is_writer = is_writer
        
        if is_writer:
            self._dict = manager.dict()
            self._init_default_values()
        else:
            self._dict = None
    
    def _init_default_values(self):
        """Initialize with default values."""
        self._dict["target_positions"] = np.zeros(NUM_MOTORS, dtype=np.float32).tolist()
        self._dict["valid"] = False
        self._dict["timestamp"] = 0.0
        
        # Individual body parts (optional, for partial commands)
        for name, indices in self.JOINT_GROUPS.items():
            self._dict[f"target_{name}"] = None
    
    def set_dict(self, shared_dict: DictProxy):
        """Set the shared dictionary (for motor worker)."""
        self._dict = shared_dict
    
    def get_dict(self) -> DictProxy:
        """Get the underlying shared dictionary."""
        return self._dict
    
    def set(self, key: str, value: np.ndarray):
        """
        Set command for a body part or full positions.
        
        Args:
            key: "positions" for all, or body part name
            value: target positions as numpy array
        """
        if key == "positions":
            self._dict["target_positions"] = value.astype(np.float32).tolist()
        elif key in self.JOINT_GROUPS:
            self._dict[f"target_{key}"] = value.astype(np.float32).tolist()
        else:
            raise KeyError(f"Unknown command key: {key}")
    
    def mark_valid(self):
        """Mark command as ready to execute."""
        # Merge body part commands into full positions
        positions = np.array(self._dict["target_positions"], dtype=np.float32)
        
        for name, indices in self.JOINT_GROUPS.items():
            target = self._dict.get(f"target_{name}")
            if target is not None:
                positions[indices] = np.array(target, dtype=np.float32)
                self._dict[f"target_{name}"] = None  # Clear after merge
        
        self._dict["target_positions"] = positions.tolist()
        self._dict["timestamp"] = time.time()
        self._dict["valid"] = True
    
    def is_valid(self) -> bool:
        """Check if there's a valid command to execute."""
        return self._dict.get("valid", False)
    
    def get_target_positions(self) -> np.ndarray:
        """Get target positions as numpy array."""
        return np.array(self._dict["target_positions"], dtype=np.float32)
    
    def mark_consumed(self):
        """Mark command as consumed (motor worker calls this)."""
        self._dict["valid"] = False


class MotorWorker:
    """
    Motor worker process for G1 robot.
    
    Handles:
    - High-frequency robot state reading
    - Command execution from shared memory
    - Joystick monitoring
    """
    
    def __init__(self, 
                 config: MotorWorkerConfig,
                 manager: Manager,
                 state_dict: DictProxy,
                 command_dict: DictProxy,
                 start_flag: Value,
                 stop_flag: Value,
                 emergency_flag: Value):
        self.config = config
        
        # Shared state/command - use provided dicts directly
        self._state_dict = state_dict
        self._command_dict = command_dict
        
        # Control flags
        self._start_flag = start_flag
        self._stop_flag = stop_flag
        self._emergency_flag = emergency_flag
        
        # Robot
        self.robot: Optional[G1RobotInterface] = None
        self._running = False
    
    def initialize(self):
        """Initialize robot connection."""
        print("[MotorWorker] Initializing robot...")
        self.robot = G1RobotInterface()
        self.robot.connect()
        print("[MotorWorker] Robot connected")
        
        # Wait for motor state DDS data (29 motors)
        print("[MotorWorker] Checking motor DDS subscription...")
        while True:
            obs = self.robot.get_observation()
            motor_positions = obs.get("motor_positions", None)
            if motor_positions is not None and len(motor_positions) == NUM_MOTORS and any(motor_positions):
                break
            time.sleep(0.01)
            print("[MotorWorker] Waiting for motor DDS subscription...")
        print(f"[MotorWorker] Motor DDS OK - receiving {NUM_MOTORS} motor states")
        
        # Wait for hand state DDS data (left and right)
        print("[MotorWorker] Checking hand DDS subscription...")
        while True:
            obs = self.robot.get_observation()
            left_hand = obs.get("left_hand_state", None)
            right_hand = obs.get("right_hand_state", None)
            if left_hand is not None and right_hand is not None:
                if any(left_hand) and any(right_hand):
                    break
            time.sleep(0.01)
            print("[MotorWorker] Waiting for hand DDS subscription...")
        print("[MotorWorker] Hand DDS OK - receiving left & right hand states")
        
        print("[MotorWorker] All DDS subscriptions ready")
    
    def run(self):
        """Main control loop."""
        self.initialize()
        self._running = True
        
        dt = 1.0 / self.config.control_frequency
        print(f"[MotorWorker] Running at {self.config.control_frequency} Hz")
        
        try:
            while self._running:
                loop_start = time.time()
                
                # 1. Read robot state
                obs = self.robot.get_observation()
                
                # 2. Update shared state dict directly
                self._update_state_dict(obs)
                
                # 3. Update control flags from joystick
                if self.robot.is_start_pressed():
                    self._start_flag.value = True
                    if self.config.verbose:
                        print("[MotorWorker] START pressed")
                
                if self.robot.is_stop_pressed():
                    self._stop_flag.value = True
                    if self.config.verbose:
                        print("[MotorWorker] STOP pressed")
                
                if self.robot.is_emergency_stop():
                    self._emergency_flag.value = True
                    print("[MotorWorker] EMERGENCY STOP!")
                
                # 4. Execute command if available
                if self.config.enable_command_execution and not self._emergency_flag.value:
                    if self._command_dict.get("valid", False):
                        target = np.array(self._command_dict["target_positions"], dtype=np.float32)
                        self.robot.send_action({"positions": target})
                        self._command_dict["valid"] = False
                        if self.config.verbose:
                            print("[MotorWorker] Command executed")
                
                # 5. Sleep for remaining time
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                elif self.config.verbose:
                    print(f"[MotorWorker] Loop overrun: {elapsed*1000:.1f}ms")
        
        except KeyboardInterrupt:
            print("[MotorWorker] Interrupted")
        finally:
            self.shutdown()
    
    def _update_state_dict(self, obs: dict):
        """Update shared state dict from observation."""
        positions = obs["motor_positions"]
        velocities = obs["motor_velocities"]
        torques = obs["motor_torques"]
        
        self._state_dict["motor_positions"] = positions.tolist()
        self._state_dict["motor_velocities"] = velocities.tolist()
        self._state_dict["motor_torques"] = torques.tolist()
        
        # Body parts
        self._state_dict["left_leg"] = positions[0:6].tolist()
        self._state_dict["right_leg"] = positions[6:12].tolist()
        self._state_dict["waist"] = positions[12:15].tolist()
        self._state_dict["left_arm"] = positions[15:22].tolist()
        self._state_dict["right_arm"] = positions[22:29].tolist()
        
        # IMU
        imu = obs.get("imu", {})
        self._state_dict["imu_quaternion"] = list(imu.get("quaternion", [0, 0, 0, 1]))
        self._state_dict["imu_rpy"] = list(imu.get("rpy", [0, 0, 0]))
        self._state_dict["imu_gyro"] = list(imu.get("gyro", [0, 0, 0]))
        self._state_dict["imu_accel"] = list(imu.get("accel", [0, 0, 0]))
        
        # Metadata
        self._state_dict["timestamp"] = time.time()
        self._state_dict["frame_count"] = self._state_dict.get("frame_count", 0) + 1
    
    def shutdown(self):
        """Clean shutdown."""
        print("[MotorWorker] Shutting down...")
        self._running = False
        if self.robot is not None:
            self.robot.disconnect()
        print("[MotorWorker] Shutdown complete")


def _run_motor_worker(config: MotorWorkerConfig,
                      state_dict: DictProxy,
                      command_dict: DictProxy,
                      start_flag: Value,
                      stop_flag: Value,
                      emergency_flag: Value):
    """Entry point for motor worker process."""
    # Note: Don't create Manager here - use dicts passed from parent
    worker = MotorWorker(
        config=config,
        manager=None,  # Not needed, dicts already created
        state_dict=state_dict,
        command_dict=command_dict,
        start_flag=start_flag,
        stop_flag=stop_flag,
        emergency_flag=emergency_flag,
    )
    worker.run()


class MotorWorkerHandle:
    """
    Handle for managing motor worker from main process.
    
    Usage:
        handle = MotorWorkerHandle()
        handle.start()
        
        # Read state
        state = handle.get_state()
        print(state["left_arm"])
        
        # Send command
        handle.set_command("left_arm", target_positions)
        handle.send_command()
        
        handle.stop()
    """
    
    def __init__(self, config: Optional[MotorWorkerConfig] = None):
        self.config = config or MotorWorkerConfig()
        
        # Manager for shared objects
        self._manager = Manager()
        
        # Shared state (motor worker writes, main reads)
        self._shared_state = SharedState(self._manager, is_writer=True)
        
        # Shared command (main writes, motor worker reads)
        self._shared_command = SharedCommand(self._manager, is_writer=True)
        
        # Control flags
        self._start_flag = Value('b', False)
        self._stop_flag = Value('b', False)
        self._emergency_flag = Value('b', False)
        
        # Process
        self._process: Optional[Process] = None
    
    def start(self):
        """Start motor worker process."""
        print("[Main] Starting motor worker...")
        
        self._process = Process(
            target=_run_motor_worker,
            args=(
                self.config,
                self._shared_state.get_dict(),
                self._shared_command.get_dict(),
                self._start_flag,
                self._stop_flag,
                self._emergency_flag,
            ),
            daemon=False,  # Cannot be daemon - needs to run independently
        )
        self._process.start()
        
        # Wait for initialization
        time.sleep(2.0)
        
        if not self._process.is_alive():
            raise RuntimeError("Motor worker failed to start")
        
        print("[Main] Motor worker started")
    
    def stop(self):
        """Stop motor worker process."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3.0)
        print("[Main] Motor worker stopped")
    
    # =========================================================================
    # State Access
    # =========================================================================
    
    def get_state(self, key: Optional[str] = None):
        """
        Get robot state.
        
        Args:
            key: Specific key like "left_arm", "right_leg", etc.
                 If None, returns full state dict.
        
        Returns:
            numpy array for specific key, or dict for all
            
        Example:
            left_arm = handle.get_state("left_arm")  # shape (7,)
            all_state = handle.get_state()  # dict with all keys
        """
        return self._shared_state.get(key)
    
    def __getitem__(self, key: str):
        """Dictionary-style access: handle["left_arm"]"""
        return self.get_state(key)
    
    # =========================================================================
    # Command Interface
    # =========================================================================
    
    def set_command(self, key: str, value: np.ndarray):
        """
        Set command for body part.
        
        Args:
            key: "positions" for all 29 motors, or body part name
            value: target positions
            
        Example:
            handle.set_command("left_arm", np.zeros(7))
            handle.set_command("right_arm", np.zeros(7))
        """
        self._shared_command.set(key, value)
    
    def send_command(self):
        """Send the command (marks it as valid for motor worker)."""
        self._shared_command.mark_valid()
    
    def send_positions(self, positions: np.ndarray):
        """Convenience: set all positions and send in one call."""
        self._shared_command.set("positions", positions)
        self._shared_command.mark_valid()
    
    # =========================================================================
    # Control Flags
    # =========================================================================
    
    @property
    def is_start_pressed(self) -> bool:
        """Check if START button was pressed."""
        if self._start_flag.value:
            self._start_flag.value = False
            return True
        return False
    
    @property
    def is_stop_pressed(self) -> bool:
        """Check if STOP button was pressed."""
        if self._stop_flag.value:
            self._stop_flag.value = False
            return True
        return False
    
    @property
    def is_emergency(self) -> bool:
        """Check if emergency stop is active."""
        return self._emergency_flag.value
    
    def wait_for_start(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for START button press.
        
        Returns:
            True if START pressed, False if timeout
        """
        print("[Main] Waiting for START button...")
        start_time = time.time()
        
        while not self._start_flag.value:
            if self._emergency_flag.value:
                raise RuntimeError("Emergency stop pressed")
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        
        self._start_flag.value = False
        print("[Main] START received!")
        return True


# ============================================================================
# Test / Debug Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Motor Worker for G1 Robot")
    parser.add_argument("--freq", type=float, default=50.0, help="Control frequency (Hz)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--test-state", action="store_true", help="Test state reading")
    args = parser.parse_args()
    
    config = MotorWorkerConfig(
        control_frequency=args.freq,
        verbose=args.verbose,
    )
    
    print("=" * 60)
    print("Motor Worker for Unitree G1")
    print("=" * 60)
    
    handle = MotorWorkerHandle(config)
    handle.start()
    
    try:
        if args.test_state:
            print("\nTest: Reading state every 0.5s")
            print("Press Ctrl+C to stop\n")
            
            while True:
                # Get full state
                state = handle.get_state()
                
                # Print body parts
                print(f"Frame {state['frame_count']:.0f}:")
                print(f"  left_arm:  {state['left_arm']}")
                print(f"  right_arm: {state['right_arm']}")
                print(f"  left_leg:  {state['left_leg']}")
                print(f"  right_leg: {state['right_leg']}")
                print(f"  waist:     {state['waist']}")
                print()
                
                time.sleep(0.5)
        else:
            print("\nMotor worker running. Press Ctrl+C to stop.\n")
            while True:
                time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        handle.stop()
