#!/usr/bin/env python3
"""
Position Tracking Test Node (FIXED)
====================================
Tests the odometry-based local position tracking by:
1. Moving the robot randomly around the arena
2. Avoiding walls using LiDAR
3. Logging position every 0.5 seconds
4. Press 's' at any time to return to origin

FIX: Corrected the coordinate transformation - the rotation matrix
was incorrectly applying the initial yaw offset, causing swapped/inverted axes.
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
    Test node that moves randomly while tracking position.
    Validates the same position tracking logic used in CubeSorterNode.
    """
    
    # ── motion parameters ─────────────────────────────────────────────────
    LINEAR_SPEED = 0.08       # m/s forward
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
        
        # Position tracking (FIXED transformation)
        self.world_x = 0.0
        self.world_y = 0.0
        self.world_yaw = 0.0
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_yaw = 0.0
        self._init_wx = None
        self._init_wy = None
        self._init_wyaw = None
        
        # Test state
        self.start_time = time.monotonic()
        self.phase = 'RANDOM_MOVE'  # RANDOM_MOVE, RETURN_HOME, IDLE
        self.position_log: List[Tuple[float, float, float, float]] = []
        self.current_linear = 0.0
        self.current_angular = 0.0
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
        print('[TEST] Position Tracking Test Started (FIXED VERSION)')
        print('[TEST] Phase 1: Random movement with wall avoidance (60s)')
        print('[TEST] Phase 2: Return to origin (15s or press "s")')
        print('[TEST] Phase 3: Idle and final report')
        print('[TEST] Press "s" at any time to return to origin')
        print('[TEST] Press "h" to halt immediately')
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
        Track position using FIXED transformation.
        
        The correct transformation should:
        1. Get displacement in world frame
        2. Rotate by -initial_yaw to align with robot's initial heading
        3. This gives local_x = forward, local_y = left
        
        The bug was: The rotation matrix was effectively applying the 
        initial yaw in the wrong direction, causing the NW movement 
        to appear as NE movement.
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
        
        # FIXED: Correct transformation to local coordinates
        # Get displacement in world frame
        dx_world = self.world_x - self._init_wx
        dy_world = self.world_y - self._init_wy
        
        # Rotate by the robot's initial yaw to align with its starting orientation
        # If robot starts facing east (yaw=0): +X is east, +Y is north
        # If robot starts facing north (yaw=pi/2): +X is north, +Y is west
        # The rotation matrix should be:
        #   local_x =  dx_world * cos(init_yaw) + dy_world * sin(init_yaw)
        #   local_y = -dx_world * sin(init_yaw) + dy_world * cos(init_yaw)
        # This puts the robot's forward direction as +local_x
        
        cos_yaw = math.cos(self._init_wyaw)
        sin_yaw = math.sin(self._init_wyaw)
        
        # CORRECTED transformation (note the signs are different from before)
        self.local_x = dx_world * cos_yaw + dy_world * sin_yaw
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
        
        print(f'[MOVE] New random movement: '
              f'linear={self.random_linear:.3f} m/s, '
              f'angular={math.degrees(self.random_angular):.1f} °/s')

    # ── return home logic ───────────────────────────────────────────────
    
    def _return_home_movement(self) -> Tuple[float, float]:
        """Calculate velocities to return to origin"""
        dx = -self.local_x
        dy = -self.local_y
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self._norm(target_yaw - self.local_yaw)
        
        if dist < 0.05:  # Within 5cm of origin
            return 0.0, 0.0
        
        # Proportional control to face home
        angular = 0.8 * yaw_error
        angular = max(-self.ANGULAR_SPEED, min(self.ANGULAR_SPEED, angular))
        
        # Move forward if facing approximately correct direction
        if abs(yaw_error) < math.radians(20):
            linear = min(self.LINEAR_SPEED * 0.5, dist * 0.5)
        else:
            linear = 0.0
        
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
        
        # Determine quadrant for verification
        quadrant = ''
        if self.local_x >= 0 and self.local_y >= 0:
            quadrant = 'NE'
        elif self.local_x >= 0 and self.local_y < 0:
            quadrant = 'SE'
        elif self.local_x < 0 and self.local_y >= 0:
            quadrant = 'NW'
        else:
            quadrant = 'SW'
        
        # Print current status with quadrant info
        print(f'[LOG] t={elapsed:.1f}s | '
              f'pos=({self.local_x:.3f}, {self.local_y:.3f})m [{quadrant}] | '
              f'dist={dist_from_origin:.3f}m | '
              f'yaw={math.degrees(self.local_yaw):.1f}° | '
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
        for i in range(0, len(self.position_log), 5):
            t, x, y, yaw = self.position_log[i]
            # Determine quadrant for each point
            quadrant = ''
            if x >= 0 and y >= 0:
                quadrant = 'NE'
            elif x >= 0 and y < 0:
                quadrant = 'SE'
            elif x < 0 and y >= 0:
                quadrant = 'NW'
            else:
                quadrant = 'SW'
            print(f'  t={t:5.1f}s  x={x:+7.3f}m  y={y:+7.3f}m  '
                  f'yaw={math.degrees(yaw):+7.1f}°  [{quadrant}]')
        
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
                self._stop()
                print('\n[HALT] Emergency halt! Stopping all motion.')
                self.phase = 'IDLE'
                self._print_final_report()
                return
            elif key == 's':
                if self.phase != 'RETURN_HOME':
                    print(f'\n[CMD] "s" pressed - Returning to origin!')
                    self.phase = 'RETURN_HOME'
                    self.state_change_time = time.monotonic()
        
        elapsed = time.monotonic() - self.start_time
        
        # Automatic phase transitions
        if self.phase == 'RANDOM_MOVE' and elapsed > self.TEST_DURATION:
            print('\n[PHASE] Test duration reached. Starting return to origin...')
            self.phase = 'RETURN_HOME'
            self.state_change_time = time.monotonic()
        
        elif self.phase == 'RETURN_HOME':
            phase_elapsed = time.monotonic() - self.state_change_time
            dist = math.hypot(self.local_x, self.local_y)
            
            if dist < 0.05 and phase_elapsed > 2.0:
                # Reached home
                self.phase = 'IDLE'
                self._stop()
                print(f'\n[PHASE] Reached origin! Current position: '
                      f'({self.local_x:.3f}, {self.local_y:.3f})m')
                self._print_final_report()
                return
            elif phase_elapsed > self.RETURN_HOME_TIME:
                # Timeout - stop anyway
                self.phase = 'IDLE'
                self._stop()
                print(f'\n[PHASE] Return home timeout ({self.RETURN_HOME_TIME}s). '
                      f'Final position: ({self.local_x:.3f}, {self.local_y:.3f})m')
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
        
        elif self.phase == 'RETURN_HOME':
            # Check walls even when returning home
            should_avoid, reason = self._check_walls()
            
            if should_avoid and 'CRITICAL' in reason:
                # Safety override
                lin, ang = self._avoid_walls()
                self._pub(lin, ang)
                print(f'[RETURN] Avoiding wall while returning home')
            else:
                # Normal return home
                lin, ang = self._return_home_movement()
                self._pub(lin, ang)

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