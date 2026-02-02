"""GR00T G1 Real Robot Evaluation - Main Entry Point

Architecture:
    Process 1 (Motor Worker): 
        - Handles G1 robot communication via DDS
        - Reads robot state at high frequency
        - Writes state to shared dict
        - Reads commands and executes them
        
    Process 2 (Main):
        - Uses TestWorker to move robot to initial position on START
"""

import time
from dataclasses import dataclass

from worker.motor_worker import MotorWorkerHandle, MotorWorkerConfig
from worker.test_worker import TestWorker, TestWorkerConfig


@dataclass
class Config:
    """Configuration for GR00T G1 evaluation."""
    control_frequency: float = 50.0
    move_duration: float = 3.0
    verbose: bool = True


def main(cfg: Config):
    """Main entry point."""
    
    motor_config = MotorWorkerConfig(
        control_frequency=cfg.control_frequency,
        verbose=cfg.verbose,
    )
    motor_worker = MotorWorkerHandle(motor_config)
    
    test_config = TestWorkerConfig(
        control_frequency=cfg.control_frequency,
        move_duration=cfg.move_duration,
        verbose=cfg.verbose,
    )
    
    try:
        # Start motor worker process
        motor_worker.start()
        
        # Run test worker (blocks until Ctrl+C)
        test_worker = TestWorker(motor_worker, test_config)
        test_worker.run()
    
    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
    
    finally:
        motor_worker.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GR00T G1 Evaluation")
    parser.add_argument("--freq", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    
    main(Config(
        control_frequency=args.freq,
        move_duration=args.duration,
        verbose=not args.quiet,
    ))
