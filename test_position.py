#!/usr/bin/env python3
"""
Position Tracking Test Node (FIXED V2 - Correct Coordinate System)
==================================================================
Tests the odometry-based local position tracking with:
- Forward = +Y, Right = +X coordinate system
- Wall avoidance using LiDAR
- Position logging every 0.5 seconds
- 's' key: return to origin
- 'h' key: return to origin then halt

Coordinate system:
  +Y (forward)
   ^
   |
   +-----> +X (right)
  Origin (0,0) at starting position
"""

import math
import time
import random
import threading
from typing import List, Tuple, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class PositionTrackerTester(Node):
    """
    Test node with corrected coordinate system:
    - Forward movement increases Y
    - Rightward movement increases X
    - Origin at starting position
    """
    
    # ── motion parameters ─────────────────────────────────────────────────
    LINEAR_SPEED = 0.16       # m/s forward
    ANGULAR_SPEED = 0.25      # rad/s rotation
    
    # ── wall avoidance ────────────────────────────────────────────────────
    WALL_WARNING_DIST = 0.40   # m - start turning away
    WALL_CRITICAL_DIST = 0.25  # m - hard stop and reverse
    WALL_SAFE_DIST = 0.50      # m - considered safe to move forward
    
    # ── timing ────────────────────────────────────────────────────────────
    CONTROL_DT = 0.05          # s - control loop period
    LOG_DT = 0.5               # s - position logging interval
    MOVE_CHANGE_DT = 2.0       # s - change random direction
    
    # ── test phases ───────────────────────────────────────────────────────
    TEST_DURATION = 60.0       # s - total test time
    RETURN_HOME_TIME = 15.0    # s - time to return to origin
    
    def __init__(self):
        super().__init__('position_tracker_tester')
        
        # ROS interfaces
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        
        # Control and logging timers
        self.create_timer(self.CONTROL_DT, self.control_loop)
        self.create_timer(self.LOG_DT, self.log_position)
        
        # Sensor data
        self.has_scan = False
        self.has_odom = False
        self.front_dist = float('inf')
        self.back_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')
        self.front_left_dist = float('inf')
        self.front_right_dist = float('inf')
        
        # Position tracking with correct coordinate system
        # Forward = +Y, Right = +X
        self.world_x = 0.0
        self.world_y = 0.0
        self.world_yaw = 0.0
        self.local_x = 0.0  # Right (+) / Left (-)
        self.local_y = 0.0  # Forward (+) / Backward (-)
        self.local_yaw = 0.0
        self._init_wx = None
        self._init_wy = None
        self._init_wyaw = None
        
        # Test state
        self.start_time = time.monotonic()
        self.phase = 'RANDOM_MOVE'  # RANDOM_MOVE, RETURN_HOME, HALT_AFTER_RETURN, IDLE
        self.position_log: List[Tuple[float, float, float, float]] = []
        self.last_move_change = time.monotonic()
        self.state_change_time = time.monotonic()
        
        # Random movement state
        self.random_linear = 0.0
        self.random_angular = 0.0
        
        # Console input thread
        self._keys: List[str] = []
        self._key_lock = threading.Lock()
        threading.Thread(target=self._console_loop, daemon=True).start()
        
        print('=' * 60)
        print('[TEST] Position Tracking Test Started (CORRECTED COORDS)')
        print('[TEST] Coordinate system: Forward = +Y, Right = +X')
        print('[TEST] Phase 1: Random movement with wall avoidance (60s)')
        print('[TEST] Phase 2: Return to origin (15s or press "s")')
        print('[TEST] Press "s" to return to origin and continue')
        print('[TEST] Press "h" to return to origin then halt')
        print('=' * 60)

    # ── console input handling ───────────────────────────────────────────
    
    def _console_loop(self):
        """Background thread for keyboard input"""
        while True:
            try:
                line = input().strip().lower()
            except EOFError:
                return
            with self._key_lock:
                self._keys.append(line)
    
    def _pop_keys(self) -> List[str]:
        """Get and clear key presses"""
        with self._key_lock:
            k, self._keys = self._keys, []
        return k

    # ── helper functions ─────────────────────────────────────────────────
    
    @staticmethod
    def _norm(a: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _q2yaw(q) -> float:
        """Convert quaternion to yaw angle"""
        return math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2))

    def _pub(self, lin: float = 0.0, ang: float = 0.0):
        """Publish velocity command"""
        m = Twist()
        m.linear.x = float(lin)
        m.angular.z = float(ang)
        self.cmd_pub.publish(m)

    def _stop(self):
        """Stop the robot"""
        self._pub(0.0, 0.0)

    # ── sensor callbacks ─────────────────────────────────────────────────
    
    def scan_cb(self, msg: LaserScan):
        """Process LiDAR data for wall detection"""
        r = msg.ranges
        n = len(r)
        
        def vmin(indices):
            vs = [r[i] for i in indices
                  if 0 <= i < n and math.isfinite(r[i]) and r[i] > 0.05]
            return min(vs) if vs else float('inf')
        
        # Multiple direction bins for better wall avoidance
        front = list(range(0, 20)) + list(range(n - 20, n))
        back = list(range(n // 2 - 20, n // 2 + 20))
        left = list(range(70, 110))
        right = list(range(250, 290))
        front_left = list(range(20, 60))
        front_right = list(range(n - 60, n - 20))
        
        self.front_dist = vmin(front)
        self.back_dist = vmin(back)
        self.left_dist = vmin(left)
        self.right_dist = vmin(right)
        self.front_left_dist = vmin(front_left)
        self.front_right_dist = vmin(front_right)
        self.has_scan = True

    def odom_cb(self, msg: Odometry):
        """
        Track position with CORRECT coordinate system:
        - Forward = +Y
        - Right = +X
        
        The robot's initial heading defines the +Y direction.
        When the robot moves forward from its starting position, Y increases.
        When the robot moves to its right from the start, X increases.
        """
        self.world_x = msg.pose.pose.position.x
        self.world_y = msg.pose.pose.position.y
        self.world_yaw = self._q2yaw(msg.pose.pose.orientation)
        
        # Initialize origin on first message
        if self._init_wx is None:
            self._init_wx = self.world_x
            self._init_wy = self.world_y
            self._init_wyaw = self.world_yaw
            print(f'[TEST] Origin set at world coords: '
                  f'({self._init_wx:.3f}, {self._init_wy:.3f}), '
                  f'yaw={math.degrees(self._init_wyaw):.1f}°')
        
        # Get displacement in world frame
        dx_world = self.world_x - self._init_wx
        dy_world = self.world_y - self._init_wy
        
        # Rotate to align with robot's initial heading
        # Forward direction (robot's initial +X in world) becomes +Y in local
        # Right direction becomes +X in local
        # 
        # Standard rotation: if robot starts facing world_yaw angle,
        # the local frame is rotated by init_yaw relative to world frame
        # 
        # Local X (right) =  world_dx * cos(yaw) + world_dy * sin(yaw)
        # Local Y (forward) = -world_dx * sin(yaw) + world_dy * cos(yaw)
        
        cos_yaw = math.cos(self._init_wyaw)
        sin_yaw = math.sin(self._init_wyaw)
        
        # X = rightward displacement from origin
        self.local_x = dx_world * cos_yaw + dy_world * sin_yaw
        # Y = forward displacement from origin  
        self.local_y = -dx_world * sin_yaw + dy_world * cos_yaw
        
        # Local yaw is relative to initial orientation
        self.local_yaw = self._norm(self.world_yaw - self._init_wyaw)
        self.has_odom = True

    # ── wall avoidance logic ─────────────────────────────────────────────
    
    def _check_walls(self) -> Tuple[bool, str]:
        """
        Check for nearby walls and determine avoidance action.
        Returns (should_avoid, reason)
        """
        # Critical: any very close wall
        if self.front_dist < self.WALL_CRITICAL_DIST:
            return True, 'FRONT_CRITICAL'
        if self.front_left_dist < self.WALL_CRITICAL_DIST:
            return True, 'FRONT_LEFT_CRITICAL'
        if self.front_right_dist < self.WALL_CRITICAL_DIST:
            return True, 'FRONT_RIGHT_CRITICAL'
        
        # Warning: approaching a wall
        if self.front_dist < self.WALL_WARNING_DIST:
            return True, 'FRONT_WARNING'
        if self.front_left_dist < self.WALL_WARNING_DIST:
            return True, 'FRONT_LEFT_WARNING'
        if self.front_right_dist < self.WALL_WARNING_DIST:
            return True, 'FRONT_RIGHT_WARNING'
        
        return False, 'CLEAR'

    def _avoid_walls(self) -> Tuple[float, float]:
        """
        Calculate avoidance velocities based on wall positions.
        Returns (linear, angular) velocities.
        """
        # Determine which direction is safest
        left_clear = self.left_dist > self.WALL_SAFE_DIST
        right_clear = self.right_dist > self.WALL_SAFE_DIST
        front_clear = self.front_dist > self.WALL_SAFE_DIST
        
        if not front_clear:
            # Front blocked - need to turn
            if left_clear and not right_clear:
                return 0.0, self.ANGULAR_SPEED  # Turn left
            elif right_clear and not left_clear:
                return 0.0, -self.ANGULAR_SPEED  # Turn right
            elif left_clear and right_clear:
                # Both clear - choose direction with more space
                if self.left_dist > self.right_dist:
                    return 0.0, self.ANGULAR_SPEED
                else:
                    return 0.0, -self.ANGULAR_SPEED
            else:
                # Trapped! Reverse and turn
                return -0.05, self.ANGULAR_SPEED * random.choice([-1, 1])
        
        # Front is clear but sides might not be
        if self.front_left_dist < self.WALL_WARNING_DIST:
            return 0.05, -self.ANGULAR_SPEED * 0.5  # Gentle right turn
        if self.front_right_dist < self.WALL_WARNING_DIST:
            return 0.05, self.ANGULAR_SPEED * 0.5  # Gentle left turn
        
        # Safe to move
        return self.random_linear, self.random_angular

    # ── random movement generation ──────────────────────────────────────
    
    def _generate_random_movement(self):
        """Generate new random movement velocities"""
        # Random forward/backward with bias toward forward
        self.random_linear = random.uniform(-0.03, self.LINEAR_SPEED)
        
        # Random rotation with bias toward smaller turns
        self.random_angular = random.gauss(0, self.ANGULAR_SPEED * 0.3)
        self.random_angular = max(-self.ANGULAR_SPEED, 
                                  min(self.ANGULAR_SPEED, self.random_angular))
        
        # Determine movement description based on velocities
        move_desc = ''
        if self.random_linear > 0.03:
            move_desc = 'forward'
        elif self.random_linear < -0.03:
            move_desc = 'backward'
        else:
            move_desc = 'stopped'
            
        if abs(self.random_angular) > 0.05:
            turn_desc = 'turning right' if self.random_angular < 0 else 'turning left'
            move_desc += f' & {turn_desc}'
        
        print(f'[MOVE] New random movement: {move_desc} '
              f'(linear={self.random_linear:.3f} m/s, '
              f'angular={math.degrees(self.random_angular):.1f} °/s)')

    # ── return home logic ───────────────────────────────────────────────
    
    def _return_home_movement(self) -> Tuple[float, float]:
        """Calculate velocities to return to origin (0,0)"""
        # Vector from current position to origin
        dx_to_home = -self.local_x  # Negative because we want to go to 0
        dy_to_home = -self.local_y
        
        dist = math.hypot(dx_to_home, dy_to_home)
        target_yaw = math.atan2(dy_to_home, dx_to_home)  # atan2(y, x) for direction to home
        
        # Current heading in local frame
        current_heading = self.local_yaw
        
        # Calculate heading error: how much we need to turn to face home
        yaw_error = self._norm(target_yaw - current_heading)
        
        if dist < 0.05:  # Within 5cm of origin
            return 0.0, 0.0
        
        # Proportional control to face home
        angular = 0.8 * yaw_error
        angular = max(-self.ANGULAR_SPEED, min(self.ANGULAR_SPEED, angular))
        
        # Move forward if facing approximately correct direction
        if abs(yaw_error) < math.radians(20):
            linear = min(self.LINEAR_SPEED * 0.5, dist * 0.5)
        else:
            linear = 0.0  # Turn in place if not facing home
        
        return linear, angular

    # ── logging ──────────────────────────────────────────────────────────
    
    def log_position(self):
        """Log current position for analysis"""
        if not self.has_odom:
            print('[LOG] Waiting for odometry...')
            return
        
        elapsed = time.monotonic() - self.start_time
        
        # Calculate distance from origin
        dist_from_origin = math.hypot(self.local_x, self.local_y)
        
        # Store for final report
        self.position_log.append((
            elapsed, self.local_x, self.local_y, self.local_yaw))
        
        # Determine cardinal direction for verification
        direction = ''
        if abs(self.local_x) < 0.05 and abs(self.local_y) < 0.05:
            direction = 'AT ORIGIN'
        else:
            if self.local_y > 0.05:
                direction += 'N'
            elif self.local_y < -0.05:
                direction += 'S'
            if self.local_x > 0.05:
                direction += 'E'
            elif self.local_x < -0.05:
                direction += 'W'
        
        # Print current status with direction
        print(f'[LOG] t={elapsed:.1f}s | '
              f'pos=(X:{self.local_x:+.3f}, Y:{self.local_y:+.3f})m [{direction}] | '
              f'dist={dist_from_origin:.3f}m | '
              f'heading={math.degrees(self.local_yaw):.1f}° | '
              f'phase={self.phase} | '
              f'walls: F={self.front_dist:.2f}m L={self.left_dist:.2f}m R={self.right_dist:.2f}m')

    def _print_final_report(self):
        """Print comprehensive test report"""
        if not self.position_log:
            print('[TEST] No position data collected!')
            return
        
        print('\n' + '=' * 60)
        print('FINAL TEST REPORT')
        print('=' * 60)
        
        # Calculate statistics
        distances = [math.hypot(x, y) for _, x, y, _ in self.position_log]
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        final_dist = distances[-1] if distances else 0
        
        # Path analysis
        if len(self.position_log) > 1:
            total_path = 0.0
            for i in range(1, len(self.position_log)):
                _, x1, y1, _ = self.position_log[i-1]
                _, x2, y2, _ = self.position_log[i]
                total_path += math.hypot(x2 - x1, y2 - y1)
        else:
            total_path = 0.0
        
        print(f'Test duration: {self.position_log[-1][0]:.1f}s')
        print(f'Number of position samples: {len(self.position_log)}')
        print(f'Maximum distance from origin: {max_dist:.3f}m')
        print(f'Average distance from origin: {avg_dist:.3f}m')
        print(f'Final distance from origin: {final_dist:.3f}m')
        print(f'Total path length (approximate): {total_path:.3f}m')
        
        # Final position check
        final_x, final_y = self.local_x, self.local_y
        print(f'Final position: X={final_x:+.3f}m, Y={final_y:+.3f}m')
        
        if final_dist < 0.10:
            print(f'\n✅ SUCCESS: Robot returned to within 10cm of origin')
        elif final_dist < 0.25:
            print(f'\n⚠️  WARNING: Robot is {final_dist*100:.0f}cm from origin '
                  f'(acceptable but not perfect)')
        else:
            print(f'\n❌ FAIL: Robot is {final_dist*100:.0f}cm from origin! '
                  f'Position tracking may have drift.')
        
        # Arena boundary check (assuming 3m arena)
        if max_dist > 3.0:
            print(f'⚠️  WARNING: Robot wandered beyond typical arena bounds '
                  f'(max distance {max_dist:.2f}m)')
        
        print('\nSample trajectory (every 5th point):')
        print('  Time    X(m)     Y(m)     Heading(°)  Direction')
        print('  ------  -------  -------  ----------  ---------')
        for i in range(0, len(self.position_log), 5):
            t, x, y, yaw = self.position_log[i]
            # Determine direction
            direction = ''
            if abs(x) < 0.05 and abs(y) < 0.05:
                direction = 'ORIGIN'
            else:
                if y > 0.05:
                    direction += 'N'
                elif y < -0.05:
                    direction += 'S'
                if x > 0.05:
                    direction += 'E'
                elif x < -0.05:
                    direction += 'W'
            print(f'  {t:5.1f}s  {x:+7.3f}  {y:+7.3f}  {math.degrees(yaw):+10.1f}  {direction:>9}')
        
        print('=' * 60)

    # ── main control loop ───────────────────────────────────────────────
    
    def control_loop(self):
        """Main control loop with state machine for testing"""
        if not self.has_scan or not self.has_odom:
            print(f'[WAIT] scan={self.has_scan} odom={self.has_odom}')
            return
        
        # Process keyboard commands
        for key in self._pop_keys():
            if key == 'h':
                # Return to origin then halt
                if self.phase not in ['RETURN_HOME', 'HALT_AFTER_RETURN']:
                    print(f'\n[CMD] "h" pressed - Returning to origin before halting...')
                    self.phase = 'HALT_AFTER_RETURN'
                    self.state_change_time = time.monotonic()
            elif key == 's':
                # Return to origin and continue (go to IDLE or resume)
                if self.phase not in ['RETURN_HOME', 'HALT_AFTER_RETURN']:
                    print(f'\n[CMD] "s" pressed - Returning to origin...')
                    self.phase = 'RETURN_HOME'
                    self.state_change_time = time.monotonic()
        
        elapsed = time.monotonic() - self.start_time
        
        # Automatic phase transitions
        if self.phase == 'RANDOM_MOVE' and elapsed > self.TEST_DURATION:
            print('\n[PHASE] Test duration reached. Starting return to origin...')
            self.phase = 'RETURN_HOME'
            self.state_change_time = time.monotonic()
        
        elif self.phase in ['RETURN_HOME', 'HALT_AFTER_RETURN']:
            dist = math.hypot(self.local_x, self.local_y)
            
            # Check if we've reached home
            if dist < 0.05:
                self._stop()
                if self.phase == 'HALT_AFTER_RETURN':
                    print(f'\n[HALT] Reached origin. Halting as requested.')
                    print(f'Final position: X={self.local_x:+.3f}m, Y={self.local_y:+.3f}m')
                    self.phase = 'IDLE'
                    self._print_final_report()
                    return
                else:
                    print(f'\n[HOME] Reached origin. Continuing operation.')
                    self.phase = 'RANDOM_MOVE'  # Go back to random movement
                    self.last_move_change = time.monotonic()
            
            # Check for timeout
            phase_elapsed = time.monotonic() - self.state_change_time
            if phase_elapsed > self.RETURN_HOME_TIME:
                self._stop()
                print(f'\n[TIMEOUT] Return home timeout ({self.RETURN_HOME_TIME}s).')
                print(f'Final position: X={self.local_x:+.3f}m, Y={self.local_y:+.3f}m')
                self.phase = 'IDLE'
                self._print_final_report()
                return
        
        elif self.phase == 'IDLE':
            self._stop()
            return
        
        # Execute current phase
        if self.phase == 'RANDOM_MOVE':
            # Generate new random movement periodically
            move_elapsed = time.monotonic() - self.last_move_change
            if move_elapsed > self.MOVE_CHANGE_DT:
                self._generate_random_movement()
                self.last_move_change = time.monotonic()
            
            # Check for walls
            should_avoid, reason = self._check_walls()
            
            if should_avoid:
                # Wall avoidance overrides random movement
                lin, ang = self._avoid_walls()
                if 'CRITICAL' in reason:
                    print(f'[AVOID] {reason}! '
                          f'FL={self.front_left_dist:.2f}m '
                          f'F={self.front_dist:.2f}m '
                          f'FR={self.front_right_dist:.2f}m')
                self._pub(lin, ang)
            else:
                # Normal random movement
                self._pub(self.random_linear, self.random_angular)
        
        elif self.phase in ['RETURN_HOME', 'HALT_AFTER_RETURN']:
            # Check walls even when returning home
            should_avoid, reason = self._check_walls()
            
            if should_avoid and 'CRITICAL' in reason:
                # Safety override for critical wall proximity
                lin, ang = self._avoid_walls()
                self._pub(lin, ang)
                print(f'[RETURN] Avoiding wall while returning home ({reason})')
            else:
                # Normal return home
                lin, ang = self._return_home_movement()
                self._pub(lin, ang)
                
                # Print progress periodically (every 2 seconds approximately)
                if int(time.monotonic() * 10) % 20 == 0:  # Every ~2 seconds
                    dist = math.hypot(self.local_x, self.local_y)
                    print(f'[RETURN] Distance to home: {dist:.2f}m | '
                          f'Heading to home: {math.degrees(math.atan2(-self.local_y, -self.local_x)):.1f}°')

    def destroy_node(self):
        """Clean shutdown"""
        self._stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PositionTrackerTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[TEST] Interrupted by user')
        node._print_final_report()
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()