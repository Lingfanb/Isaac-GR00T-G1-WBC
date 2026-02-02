#!/usr/bin/env python3
"""
G1 Interface Test Script

Tests all components:
- 29 body joints (position reading)
- IMU data
- 14 hand fingers (Dex3-1)
- Joystick buttons and axes
"""

import time
import sys
import numpy as np

# Add parent path for imports
sys.path.insert(0, "/home/lingfanb/Lingfan_Research_UCL/Isaac-GR00T")


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def print_separator(title: str):
    """Print section separator."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def test_body_joints(robot):
    """Test 29 body joint readings."""
    print_separator("BODY JOINTS (29 motors)")
    
    positions = robot._state_receiver.get_motor_positions()
    velocities = robot._state_receiver.get_motor_velocities()
    
    joint_names = [
        # Legs
        "L_HipPitch", "L_HipRoll", "L_HipYaw", "L_Knee", "L_AnkleP", "L_AnkleR",
        "R_HipPitch", "R_HipRoll", "R_HipYaw", "R_Knee", "R_AnkleP", "R_AnkleR",
        # Waist
        "WaistYaw", "WaistRoll", "WaistPitch",
        # Left Arm
        "L_ShoulderP", "L_ShoulderR", "L_ShoulderY", "L_Elbow", "L_WristR", "L_WristP", "L_WristY",
        # Right Arm  
        "R_ShoulderP", "R_ShoulderR", "R_ShoulderY", "R_Elbow", "R_WristR", "R_WristP", "R_WristY",
    ]
    
    print(f"{'Index':<6} {'Joint':<14} {'Pos (rad)':<12} {'Pos (deg)':<12} {'Vel':<10}")
    print("-" * 60)
    
    for i, name in enumerate(joint_names):
        pos_rad = positions[i]
        pos_deg = np.rad2deg(pos_rad)
        vel = velocities[i]
        print(f"{i:<6} {name:<14} {pos_rad:>10.4f} {pos_deg:>10.2f}° {vel:>10.4f}")


def test_imu(robot):
    """Test IMU data."""
    print_separator("IMU DATA")
    
    imu = robot._state_receiver.get_imu_data()
    
    print(f"Quaternion: [{imu['quaternion'][0]:.4f}, {imu['quaternion'][1]:.4f}, "
          f"{imu['quaternion'][2]:.4f}, {imu['quaternion'][3]:.4f}]")
    print(f"RPY (deg):  Roll={np.rad2deg(imu['rpy'][0]):.2f}°, "
          f"Pitch={np.rad2deg(imu['rpy'][1]):.2f}°, "
          f"Yaw={np.rad2deg(imu['rpy'][2]):.2f}°")
    print(f"Gyroscope:  [{imu['gyro'][0]:.4f}, {imu['gyro'][1]:.4f}, {imu['gyro'][2]:.4f}] rad/s")
    print(f"Accel:      [{imu['accel'][0]:.4f}, {imu['accel'][1]:.4f}, {imu['accel'][2]:.4f}] m/s²")


def test_hands(hands):
    """Test 14 hand finger joints (7 per hand)."""
    print_separator("DEX3-1 HANDS (14 joints)")
    
    left_pos = hands.get_left_positions()
    right_pos = hands.get_right_positions()
    
    finger_names = ["Thumb", "Index1", "Index2", "Middle", "Ring", "Pinky1", "Pinky2"]
    
    print(f"{'Finger':<12} {'Left (rad)':<12} {'Left (deg)':<12} {'Right (rad)':<12} {'Right (deg)':<12}")
    print("-" * 60)
    
    for i, name in enumerate(finger_names):
        l_rad = left_pos[i]
        l_deg = np.rad2deg(l_rad)
        r_rad = right_pos[i]
        r_deg = np.rad2deg(r_rad)
        print(f"{name:<12} {l_rad:>10.4f} {l_deg:>10.2f}° {r_rad:>10.4f} {r_deg:>10.2f}°")


def test_joystick(robot):
    """Test joystick buttons and axes."""
    print_separator("JOYSTICK / REMOTE CONTROLLER")
    
    robot._update_joystick()
    js = robot.joystick
    
    # Axes
    print("AXES:")
    print(f"  Left Stick:  X={js.Lx:>6.3f}  Y={js.Ly:>6.3f}")
    print(f"  Right Stick: X={js.Rx:>6.3f}  Y={js.Ry:>6.3f}")
    
    # Buttons
    print("\nBUTTONS:")
    buttons = [
        ("L1", js.L1), ("L2", js.L2), ("R1", js.R1), ("R2", js.R2),
        ("A", js.A), ("B", js.B), ("X", js.X), ("Y", js.Y),
        ("Up", js.Up), ("Down", js.Down), ("Left", js.Left), ("Right", js.Right),
        ("Start", js.Start), ("Select", js.Select), ("F1", js.F1), ("F3", js.F3),
    ]
    
    row1 = " | ".join([f"{name}:{'▣' if val else '▢'}" for name, val in buttons[:8]])
    row2 = " | ".join([f"{name}:{'▣' if val else '▢'}" for name, val in buttons[8:]])
    print(f"  {row1}")
    print(f"  {row2}")
    
    # Locomotion command
    vx, vy, vyaw = js.get_locomotion_command()
    print(f"\nLocomotion: vx={vx:.3f}, vy={vy:.3f}, vyaw={vyaw:.3f}")


def run_zero_position_test(robot, hands=None):
    """Test moving all upper body joints to zero position.
    
    Includes: Arms (14), Waist (3), and optionally Hands (14).
    Legs are kept at current position for safety.
    """
    print("\n" + "="*60)
    print(" FULL BODY ZERO POSITION TEST")
    print("="*60)
    
    print("\n⚠ WARNING: This will move the robot's upper body!")
    print("  - Arms (14 joints): Will move to zero")
    print("  - Waist (3 joints): Will move to zero")
    print("  - Hands (14 joints): Will move to zero" if hands else "  - Hands: Skipped")
    print("  - Legs (12 joints): KEPT AT CURRENT POSITION (safety)")
    print("\nMake sure the robot has clearance.")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Get current positions
    current_positions = robot._state_receiver.get_motor_positions()
    
    # Joint indices (G1 configuration)
    # Legs: 0-11 (12 joints)
    # Waist: 12-14 (3 joints)
    # Left Arm: 15-21 (7 joints)
    # Right Arm: 22-28 (7 joints)
    
    LEG_INDICES = list(range(0, 12))    # Keep current
    WAIST_INDICES = list(range(12, 15)) # Move to zero
    LEFT_ARM_INDICES = list(range(15, 22))   # Move to zero
    RIGHT_ARM_INDICES = list(range(22, 29))  # Move to zero
    
    print("\n--- Current Positions ---")
    print(f"Waist (12-14):     {np.round(current_positions[12:15], 3)}")
    print(f"Left Arm (15-21):  {np.round(current_positions[15:22], 3)}")
    print(f"Right Arm (22-28): {np.round(current_positions[22:29], 3)}")
    
    if hands:
        current_left_hand = hands.get_left_positions()
        current_right_hand = hands.get_right_positions()
        print(f"Left Hand:  {np.round(current_left_hand, 3)}")
        print(f"Right Hand: {np.round(current_right_hand, 3)}")
    
    # Build target positions
    target_positions = current_positions.copy()
    target_positions[WAIST_INDICES] = 0.0
    target_positions[LEFT_ARM_INDICES] = 0.0
    target_positions[RIGHT_ARM_INDICES] = 0.0
    # Legs stay at current position
    
    target_left_hand = np.zeros(7) if hands else None
    target_right_hand = np.zeros(7) if hands else None
    
    # Smooth interpolation over 3 seconds
    duration = 3.0
    steps = 100
    dt = duration / steps
    
    print(f"\nMoving to zero position over {duration}s...")
    
    for i in range(steps + 1):
        t = i / steps  # 0 to 1
        # Smooth interpolation (ease in-out)
        smooth_t = t * t * (3 - 2 * t)
        
        # Interpolate body positions
        interp_positions = current_positions + smooth_t * (target_positions - current_positions)
        
        # Send full body command
        robot.send_full_body_action(interp_positions)
        
        # Interpolate hands
        if hands:
            left_hand = current_left_hand + smooth_t * (target_left_hand - current_left_hand)
            right_hand = current_right_hand + smooth_t * (target_right_hand - current_right_hand)
            hands.send_both_commands(left_hand, right_hand)
        
        time.sleep(dt)
    
    print("✓ Reached zero position!")
    
    # Show final positions
    final_positions = robot._state_receiver.get_motor_positions()
    print("\n--- Final Positions ---")
    print(f"Waist (12-14):     {np.round(final_positions[12:15], 3)}")
    print(f"Left Arm (15-21):  {np.round(final_positions[15:22], 3)}")
    print(f"Right Arm (22-28): {np.round(final_positions[22:29], 3)}")
    
    if hands:
        final_left_hand = hands.get_left_positions()
        final_right_hand = hands.get_right_positions()
        print(f"Left Hand:  {np.round(final_left_hand, 3)}")
        print(f"Right Hand: {np.round(final_right_hand, 3)}")


def run_continuous_test(robot, hands):
    """Run continuous test loop."""
    print("\n" + "="*60)
    print(" CONTINUOUS TEST MODE - Press Ctrl+C to exit")
    print("="*60)
    
    try:
        while True:
            clear_screen()
            print(f"G1 Interface Test - {time.strftime('%H:%M:%S')}")
            
            test_body_joints(robot)
            test_imu(robot)
            
            if hands is not None:
                test_hands(hands)
            
            test_joystick(robot)
            
            print("\n[Press Ctrl+C to exit]")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")


def run_single_test(robot, hands):
    """Run single test."""
    print("\n" + "="*60)
    print(" G1 Interface Single Test")
    print("="*60)
    
    test_body_joints(robot)
    test_imu(robot)
    
    if hands is not None:
        test_hands(hands)
    
    test_joystick(robot)


def main():
    """Main test entry point."""
    from UnitreeG1Interface import G1RobotInterface
    from UnitreeG1Interface.g1_hands import Dex3HandInterface
    
    print("="*60)
    print(" G1 Interface Test")
    print("="*60)
    
    # Parse arguments
    continuous = "--continuous" in sys.argv or "-c" in sys.argv
    skip_hands = "--no-hands" in sys.argv
    test_zero = "--test-zero" in sys.argv
    
    print("\nInitializing G1 Robot Interface...")
    robot = G1RobotInterface()
    
    try:
        robot.connect()
        print("✓ Robot connected")
        
        # Initialize hands
        hands = None
        if not skip_hands:
            print("\nInitializing Dex3-1 Hands...")
            try:
                hands = Dex3HandInterface()
                hands.init()
                if hands.wait_for_state(timeout=2.0):
                    print("✓ Hands connected")
                else:
                    print("✗ Hands timeout - running without hands")
                    hands = None
            except Exception as e:
                print(f"✗ Hands failed: {e} - running without hands")
                hands = None
        
        # Run test
        if test_zero:
            run_zero_position_test(robot, hands)
        elif continuous:
            run_continuous_test(robot, hands)
        else:
            run_single_test(robot, hands)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
    finally:
        robot.disconnect()
        print("\n✓ Robot disconnected")


if __name__ == "__main__":
    print("Usage: python test_interface.py [--continuous/-c] [--no-hands] [--test-zero]")
    print("  --continuous, -c : Run continuous test loop")
    print("  --no-hands       : Skip hand testing")
    print("  --test-zero      : Test moving arms to zero position")
    print()
    main()
