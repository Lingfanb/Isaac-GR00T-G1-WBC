#!/usr/bin/env python3
"""
Motor Worker Launcher

单独启动 Motor Worker 进程，用于与 G1 机器人通信。
其他进程（如 test_worker, groot_worker）可以通过共享内存与之通信。

Usage:
    python run_motor_worker.py --freq 50 --verbose
"""

import argparse
import signal
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from worker.motor_worker import MotorWorkerHandle, MotorWorkerConfig


def main():
    parser = argparse.ArgumentParser(description="Motor Worker for G1 Robot")
    parser.add_argument("--freq", type=float, default=50.0, help="Control frequency (Hz)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    config = MotorWorkerConfig(
        control_frequency=args.freq,
        verbose=args.verbose,
    )
    
    print("=" * 60)
    print("Motor Worker Launcher")
    print("=" * 60)
    print(f"  Control Frequency: {config.control_frequency} Hz")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    # Create and start motor worker
    motor_worker = MotorWorkerHandle(config)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[Launcher] Shutting down...")
        motor_worker.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    motor_worker.start()
    
    # Keep running and print status periodically
    try:
        import time
        while True:
            state = motor_worker.get_state()
            frame = state.get("frame_count", 0)
            print(f"\r[Motor Worker] Running... Frame: {frame:.0f}", end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        motor_worker.stop()


if __name__ == "__main__":
    main()
