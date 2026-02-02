"""
Test Worker - Move robot to initial position (all zeros)

这是一个测试 worker，用于：
1. 从 Motor Worker 读取机器人状态
2. 按下 START 后，平滑地将机器人移动到初始位置 (0,0,0,0,0...)
3. 显示所有状态信息

Usage:
    # 首先在另一个终端启动 motor worker:
    python run_motor_worker.py
    
    # 然后运行 test worker:
    python -m worker.test_worker
"""

import time
import numpy as np
from multiprocessing import Process, Value
from multiprocessing.managers import DictProxy
from typing import Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from worker.motor_worker import SharedState, SharedCommand, MotorWorkerHandle, MotorWorkerConfig


@dataclass  
class TestWorkerConfig:
    """Test worker configuration."""
    control_frequency: float = 50.0  # Hz
    move_duration: float = 3.0       # Seconds to reach target
    verbose: bool = True


class TestWorker:
    """
    Test worker that moves robot to initial position.
    
    状态:
        IDLE -> 等待 START 按钮
        MOVING -> 正在移动到目标位置
        DONE -> 移动完成
    """
    
    # Initial position: all zeros for arms (adjust as needed)
    TARGET_POSITIONS = {
        "left_arm": np.zeros(7, dtype=np.float32),
        "right_arm": np.zeros(7, dtype=np.float32),
        # Note: legs and waist usually controlled by locomotion, not by us
    }
    
    def __init__(self, motor_worker: MotorWorkerHandle, config: TestWorkerConfig):
        self.motor_worker = motor_worker
        self.config = config
        self.state = "IDLE"
        
        # Movement interpolation
        self._start_positions = {}
        self._move_start_time = 0.0
    
    def run(self):
        """Main loop."""
        dt = 1.0 / self.config.control_frequency
        
        print("\n" + "=" * 60)
        print("Test Worker - Move to Initial Position")
        print("=" * 60)
        print("\n目标位置:")
        for name, pos in self.TARGET_POSITIONS.items():
            print(f"  {name}: {pos}")
        print("\n按下遥控器 START 按钮开始移动")
        print("按 Ctrl+C 退出\n")
        
        try:
            while True:
                loop_start = time.time()
                
                # 1. 读取状态
                state = self.motor_worker.get_state()
                
                # 2. 打印状态
                if self.config.verbose and int(state["frame_count"]) % 25 == 0:
                    self._print_state(state)
                
                # 3. 状态机
                if self.state == "IDLE":
                    self._handle_idle()
                    
                elif self.state == "MOVING":
                    self._handle_moving(state)
                    
                elif self.state == "DONE":
                    self._handle_done()
                
                # 4. Check emergency
                if self.motor_worker.is_emergency:
                    print("\n[TestWorker] EMERGENCY STOP!")
                    break
                
                # 5. Sleep
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n[TestWorker] Interrupted")
    
    def _print_state(self, state: dict):
        """Print robot state."""
        print("-" * 40)
        print(f"Frame: {state['frame_count']:.0f}")
        print(f"  left_arm:  {np.round(state['left_arm'], 3)}")
        print(f"  right_arm: {np.round(state['right_arm'], 3)}")
        print(f"  left_leg:  {np.round(state['left_leg'], 3)}")
        print(f"  right_leg: {np.round(state['right_leg'], 3)}")
        print(f"  waist:     {np.round(state['waist'], 3)}")
        print(f"  IMU rpy:   {np.round(state['imu_rpy'], 3)}")
    
    def _handle_idle(self):
        """Wait for START button."""
        if self.motor_worker.is_start_pressed:
            print("\n[TestWorker] START pressed! Beginning movement...")
            
            # Save current positions
            state = self.motor_worker.get_state()
            for name in self.TARGET_POSITIONS.keys():
                self._start_positions[name] = state[name].copy()
            
            self._move_start_time = time.time()
            self.state = "MOVING"
    
    def _handle_moving(self, state: dict):
        """Interpolate to target position."""
        elapsed = time.time() - self._move_start_time
        ratio = min(1.0, elapsed / self.config.move_duration)
        
        # Smooth ease-in-out
        ratio = self._ease_in_out(ratio)
        
        # Interpolate each body part
        for name, target in self.TARGET_POSITIONS.items():
            start = self._start_positions[name]
            interpolated = (1.0 - ratio) * start + ratio * target
            self.motor_worker.set_command(name, interpolated)
        
        # Send command
        self.motor_worker.send_command()
        
        # Progress
        progress = int(ratio * 100)
        print(f"\r[TestWorker] Moving... {progress}%", end="", flush=True)
        
        if ratio >= 1.0:
            print("\n[TestWorker] Movement complete!")
            self.state = "DONE"
    
    def _handle_done(self):
        """Movement complete, wait for next START."""
        # Check for another START to repeat
        if self.motor_worker.is_start_pressed:
            print("\n[TestWorker] Repeating movement...")
            state = self.motor_worker.get_state()
            for name in self.TARGET_POSITIONS.keys():
                self._start_positions[name] = state[name].copy()
            self._move_start_time = time.time()
            self.state = "MOVING"
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Smooth ease-in-out interpolation."""
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - (-2 * t + 2) ** 2 / 2


def run_test_worker(motor_worker: MotorWorkerHandle, config: TestWorkerConfig):
    """Entry point for test worker process."""
    worker = TestWorker(motor_worker, config)
    worker.run()


def main():
    """Main entry point - starts both motor worker and test worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Worker")
    parser.add_argument("--freq", type=float, default=50.0, help="Control frequency")
    parser.add_argument("--duration", type=float, default=3.0, help="Move duration (seconds)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    # Motor worker config
    motor_config = MotorWorkerConfig(
        control_frequency=args.freq,
        verbose=False,
    )
    
    # Test worker config
    test_config = TestWorkerConfig(
        control_frequency=args.freq,
        move_duration=args.duration,
        verbose=not args.quiet,
    )
    
    # Create motor worker handle
    motor_worker = MotorWorkerHandle(motor_config)
    
    try:
        # Start motor worker
        motor_worker.start()
        
        # Run test worker in main process
        run_test_worker(motor_worker, test_config)
    
    finally:
        motor_worker.stop()


if __name__ == "__main__":
    main()
