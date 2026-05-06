#!/usr/bin/env python3
"""
TurtleBot3 – Red Cube Detection, Approach & Pickup (Fixed)
------------------------------------------------------------
- Searches for a red cube using the camera
- Uses a 200×480 center crop as the area of interest
- Adjusts rotation to center the cube within the crop
- Performs a full rotation if the cube is lost
- Approaches using relative size comparison to estimate distance
- Lowers servo to grab the cube when close enough
"""

import math
import time
import threading

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

# Servo control
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print('[WARN] RPi.GPIO not available – servo control disabled')


class RedCubeApproachNode(Node):
    """
    Detects a red cube, centers it in a 200×480 region of interest,
    approaches while monitoring size change, and grabs when close.
    """

    # ── Camera parameters ────────────────────────────────────────────────────
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    ROI_WIDTH = 200
    ROI_HEIGHT = 480

    # ── Rotation control ─────────────────────────────────────────────────────
    SEARCH_ANGULAR_SPEED = 0.3       # rad/s – constant speed during full rotation
    CENTER_ANGULAR_MIN = 0.06        # rad/s – slowest centering rotation
    CENTER_ANGULAR_MAX = 0.25        # rad/s – fastest centering rotation
    CENTER_DEADZONE = 8              # pixels – consider "centered" within ± this
    CENTER_KP = 0.003                # P-gain: angular_vel = KP * pixel_error

    # ── Approach control ─────────────────────────────────────────────────────
    APPROACH_SPEED = 0.08            # m/s forward
    APPROACH_ANGULAR_KP = 0.002      # milder centering while approaching
    GRAB_AREA_RATIO = 3.0            # grab when current_area >= initial_area * this

    # ── Detection parameters ─────────────────────────────────────────────────
    MIN_CONTOUR_AREA = 300
    MIN_BBOX_WIDTH = 15
    MIN_BBOX_HEIGHT = 15
    MIN_SOLIDITY = 0.70
    MIN_FILL_RATIO = 0.22
    COLOR_CONFIDENCE_MIN = 0.60

    # ── Safety ───────────────────────────────────────────────────────────────
    MIN_FRONT_CLEARANCE = 0.16       # m – stop approach if front is closer
    EMERGENCY_CLEARANCE = 0.10       # m – stop everything

    # ── Timing ───────────────────────────────────────────────────────────────
    CONTROL_RATE = 0.05              # 20 Hz control loop

    # ── Servo ────────────────────────────────────────────────────────────────
    SERVO_PIN = 12
    SERVO_UP_DUTY = 5.5
    SERVO_DOWN_DUTY = 9.0
    SERVO_FREQ = 50
    SERVO_GRAB_HOLD = 1.0            # seconds to hold servo down

    def __init__(self):
        super().__init__('red_cube_approach')

        # ── Publishers & Subscribers ─────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.create_subscription(
            LaserScan, '/scan',
            self._scan_callback,
            qos_profile_sensor_data)
        
        self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self._image_callback,
            qos_profile_sensor_data)

        # ── Servo initialization ─────────────────────────────────────────────
        self._servo_pwm = None
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.SERVO_PIN, GPIO.OUT)
                self._servo_pwm = GPIO.PWM(self.SERVO_PIN, self.SERVO_FREQ)
                self._servo_pwm.start(self.SERVO_UP_DUTY)
                self.get_logger().info(f'Servo ready on GPIO {self.SERVO_PIN}')
            except Exception as e:
                self.get_logger().error(f'Servo init failed: {e}')

        # ── Sensor data (protected by lock where needed) ─────────────────────
        self._image_lock = threading.Lock()
        self._latest_frame = None
        self._detection_result = None   # dict or None
        self._has_image = False
        
        self._front_dist = float('inf')
        self._left_dist = float('inf')
        self._right_dist = float('inf')
        self._back_dist = float('inf')
        self._has_lidar = False

        # ── State variables ──────────────────────────────────────────────────
        self._state = 'WAITING'
        self._state_start = time.monotonic()
        self._initial_bbox_area = 0.0
        self._search_angle_accumulated = 0.0  # radians rotated during search

        # ── Morphology kernel ────────────────────────────────────────────────
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # ── Control timer ────────────────────────────────────────────────────
        self._control_timer = self.create_timer(self.CONTROL_RATE, self._control_loop)

        # ── Console thread ───────────────────────────────────────────────────
        self._console_thread = threading.Thread(target=self._console_watcher, daemon=True)
        self._console_thread.start()

        self.get_logger().info('=== Red Cube Approach Node ===')
        self.get_logger().info('Press ENTER to start searching.')

    # ═══════════════════════════════════════════════════════════════════════════
    # Servo
    # ═══════════════════════════════════════════════════════════════════════════

    def _servo_lift(self):
        self.get_logger().info('SERVO: lifting')
        if self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(self.SERVO_UP_DUTY)

    def _servo_lower(self):
        self.get_logger().info('SERVO: lowering (GRAB)')
        if self._servo_pwm:
            self._servo_pwm.ChangeDutyCycle(self.SERVO_DOWN_DUTY)

    # ═══════════════════════════════════════════════════════════════════════════
    # LiDAR callback
    # ═══════════════════════════════════════════════════════════════════════════

    def _scan_callback(self, msg: LaserScan):
        ranges = msg.ranges
        n = len(ranges)
        if n < 360:
            return

        def sector_min(center_idx, spread):
            idx = []
            for i in range(center_idx - spread, center_idx + spread + 1):
                idx.append(i % n)
            vals = [ranges[i] for i in idx if math.isfinite(ranges[i]) and ranges[i] > 0.03]
            return min(vals) if vals else float('inf')

        # Front: 0°, Back: 180°, Left: 90°, Right: 270°
        raw_front = sector_min(0, 12)
        raw_back = sector_min(n // 2, 12)
        raw_left = sector_min(n // 4, 12)
        raw_right = sector_min(3 * n // 4, 12)

        alpha = 0.5
        if not self._has_lidar:
            self._front_dist = raw_front
            self._back_dist = raw_back
            self._left_dist = raw_left
            self._right_dist = raw_right
            self._has_lidar = True
        else:
            self._front_dist = alpha * self._front_dist + (1 - alpha) * raw_front
            self._back_dist = alpha * self._back_dist + (1 - alpha) * raw_back
            self._left_dist = alpha * self._left_dist + (1 - alpha) * raw_left
            self._right_dist = alpha * self._right_dist + (1 - alpha) * raw_right

    # ═══════════════════════════════════════════════════════════════════════════
    # Image callback – only does detection, stores result
    # ═══════════════════════════════════════════════════════════════════════════

    def _image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (self.IMG_HEIGHT, self.IMG_WIDTH):
                return
        except Exception:
            return

        detection = self._find_red_cube(frame)

        with self._image_lock:
            self._latest_frame = frame
            self._has_image = True
            self._detection_result = detection

    # ═══════════════════════════════════════════════════════════════════════════
    # Red cube detection (robust)
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_red_cube(self, frame):
        """
        Returns dict with keys: cx, cy, area, bbox_w, bbox_h, roi_center
        or None if no valid red cube is found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red hue wraps around 0
        lower_red1 = np.array([0, 85, 35])
        upper_red1 = np.array([14, 255, 255])
        lower_red2 = np.array([166, 85, 35])
        upper_red2 = np.array([180, 255, 255])

        mask_hsv = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2))

        # BGR verification: red channel must dominate
        b, g, r = cv2.split(frame)
        mask_bgr = np.zeros(mask_hsv.shape, dtype=np.uint8)
        mask_bgr[(r >= 60) & (r > g + 15) & (r > b + 15)] = 255

        combined = cv2.bitwise_and(mask_hsv, mask_bgr)

        # Morphological operations
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < self.MIN_BBOX_WIDTH or h < self.MIN_BBOX_HEIGHT:
                continue

            # Approximate square
            aspect = w / float(h)
            if not (0.55 <= aspect <= 1.85):
                continue

            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity < self.MIN_SOLIDITY:
                continue

            # Fill ratio
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = cnt - np.array([[x, y]])
            cv2.drawContours(roi_mask, [shifted], -1, 255, -1)
            fill = np.count_nonzero(roi_mask) / float(w * h)
            if fill < self.MIN_FILL_RATIO:
                continue

            # Color confidence inside contour
            roi_hsv = hsv[y:y + h, x:x + w]
            roi_mask_bool = roi_mask > 0
            hue_vals = roi_hsv[:, :, 0][roi_mask_bool]
            sat_vals = roi_hsv[:, :, 1][roi_mask_bool]

            if len(hue_vals) < 10:
                continue

            red_hue_ratio = np.mean((hue_vals <= 14) | (hue_vals >= 166))
            high_sat_ratio = np.mean(sat_vals >= 100)
            confidence = red_hue_ratio * high_sat_ratio

            if confidence < self.COLOR_CONFIDENCE_MIN:
                continue

            if area > best_area:
                best_area = area
                cx = x + w / 2.0
                cy = y + h / 2.0
                best = {
                    'cx': cx,
                    'cy': cy,
                    'area': area,
                    'bbox_w': w,
                    'bbox_h': h,
                }

        return best

    # ═══════════════════════════════════════════════════════════════════════════
    # ROI check
    # ═══════════════════════════════════════════════════════════════════════════

    def _in_roi(self, detection):
        """Return True if the detection's cx is within the center 200px band."""
        roi_left = (self.IMG_WIDTH - self.ROI_WIDTH) // 2   # 220
        roi_right = roi_left + self.ROI_WIDTH                # 420
        return roi_left <= detection['cx'] <= roi_right

    # ═══════════════════════════════════════════════════════════════════════════
    # Motion helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _move(self, linear=0.0, angular=0.0):
        """Publish a velocity command."""
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self._cmd_pub.publish(twist)

    def _halt(self):
        self._move(0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Safety
    # ═══════════════════════════════════════════════════════════════════════════

    def _any_direction_unsafe(self):
        if not self._has_lidar:
            return False
        return (self._front_dist < self.EMERGENCY_CLEARANCE or
                self._left_dist < self.EMERGENCY_CLEARANCE or
                self._right_dist < self.EMERGENCY_CLEARANCE or
                self._back_dist < self.EMERGENCY_CLEARANCE)

    def _front_blocked(self):
        if not self._has_lidar:
            return False
        return self._front_dist < self.MIN_FRONT_CLEARANCE

    # ═══════════════════════════════════════════════════════════════════════════
    # Console thread – press Enter to start
    # ═══════════════════════════════════════════════════════════════════════════

    def _console_watcher(self):
        while True:
            try:
                input()
                if self._state == 'WAITING':
                    self.get_logger().info('>>> Starting search <<<')
                    self._state = 'ROTATE_SEARCH'
                    self._state_start = time.monotonic()
                    self._search_angle_accumulated = 0.0
                    self._initial_bbox_area = 0.0
            except EOFError:
                break

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN CONTROL LOOP – called at 20 Hz
    # ═══════════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        now = time.monotonic()

        # ═══════════════════════════════════════════════════════════════════════
        # Global safety check – override everything
        # ═══════════════════════════════════════════════════════════════════════
        if self._any_direction_unsafe():
            self._halt()
            self._state = 'WAITING'
            self.get_logger().warn('EMERGENCY – obstacle too close.')
            return

        # ═══════════════════════════════════════════════════════════════════════
        # Get latest detection snapshot (thread-safe)
        # ═══════════════════════════════════════════════════════════════════════
        with self._image_lock:
            detected = self._detection_result is not None
            detection = self._detection_result.copy() if detected else None

        # ═══════════════════════════════════════════════════════════════════════
        # WAITING
        # ═══════════════════════════════════════════════════════════════════════
        if self._state == 'WAITING':
            self._halt()
            return

        # ═══════════════════════════════════════════════════════════════════════
        # ROTATE_SEARCH – full 360° rotation looking for any red cube
        # ═══════════════════════════════════════════════════════════════════════
        if self._state == 'ROTATE_SEARCH':
            dt = self.CONTROL_RATE
            self._search_angle_accumulated += self.SEARCH_ANGULAR_SPEED * dt

            # Check if we've completed a full rotation
            if self._search_angle_accumulated >= 2.0 * math.pi:
                self._halt()
                self.get_logger().info('Full rotation done – no cube found.')
                self._state = 'WAITING'
                return

            # Check if we see a cube
            if detected and detection is not None:
                self._halt()
                self._initial_bbox_area = detection['area']
                self.get_logger().info(
                    f'Cube detected! Area = {self._initial_bbox_area:.0f} px². Centering...')
                self._state = 'CENTERING'
                self._state_start = now
                return

            # Keep rotating
            self._move(0.0, self.SEARCH_ANGULAR_SPEED)
            return

        # ═══════════════════════════════════════════════════════════════════════
        # CENTERING – rotate to bring cube into the ROI
        # ═══════════════════════════════════════════════════════════════════════
        if self._state == 'CENTERING':
            # Timeout after 8 seconds – cube might be gone
            if now - self._state_start > 8.0:
                self._halt()
                self.get_logger().info('Centering timeout – restarting search rotation.')
                self._state = 'ROTATE_SEARCH'
                self._search_angle_accumulated = 0.0
                self._state_start = now
                return

            # Lost the cube – go back to rotation search
            if not detected or detection is None:
                self._halt()
                self.get_logger().info('Cube lost during centering – resuming rotation.')
                self._state = 'ROTATE_SEARCH'
                self._search_angle_accumulated = 0.0
                self._state_start = now
                return

            # Check if centered
            if self._in_roi(detection):
                # Also check if we are close to image center (within deadzone)
                error = abs(detection['cx'] - self.IMG_WIDTH / 2.0)
                if error <= self.CENTER_DEADZONE:
                    self._halt()
                    self.get_logger().info('Cube centered in ROI. Approaching.')
                    self._state = 'APPROACHING'
                    self._state_start = now
                    return
                else:
                    # Within ROI but not dead-center – fine-tune
                    pixel_err = detection['cx'] - self.IMG_WIDTH / 2.0
                    angular = -self.CENTER_KP * pixel_err
                    angular = max(self.CENTER_ANGULAR_MIN,
                                  min(self.CENTER_ANGULAR_MAX, abs(angular)))
                    angular = math.copysign(angular, -pixel_err)
                    self._move(0.0, angular)
                    return

            # Cube visible but outside ROI – rotate toward it
            pixel_err = detection['cx'] - self.IMG_WIDTH / 2.0
            angular = -self.CENTER_KP * pixel_err
            angular = max(self.CENTER_ANGULAR_MIN,
                          min(self.CENTER_ANGULAR_MAX, abs(angular)))
            angular = math.copysign(angular, -pixel_err)
            self._move(0.0, angular)
            return

        # ═══════════════════════════════════════════════════════════════════════
        # APPROACHING – drive forward, keep centered, monitor size
        # ═══════════════════════════════════════════════════════════════════════
        if self._state == 'APPROACHING':
            # Front blocked by obstacle
            if self._front_blocked():
                self._halt()
                self.get_logger().info('Front blocked – grabbing now.')
                self._state = 'GRAB'
                self._state_start = now
                return

            # Lost the cube – assume it slipped underneath
            if not detected or detection is None:
                self._halt()
                # Only assume underneath if we had been seeing it for a while
                if now - self._state_start > 1.5:
                    self.get_logger().info('Cube lost during approach – likely underneath. Grabbing.')
                    self._state = 'GRAB'
                    self._state_start = now
                else:
                    self.get_logger().info('Cube lost early – restarting search.')
                    self._state = 'ROTATE_SEARCH'
                    self._search_angle_accumulated = 0.0
                    self._state_start = now
                return

            # Check size ratio for grabbing
            area_ratio = detection['area'] / max(self._initial_bbox_area, 1.0)
            
            # Log progress
            if int(now * 2) % 4 == 0:  # ~ every 2 seconds
                self.get_logger().info(
                    f'Approaching – area={detection["area"]:.0f}  '
                    f'ratio={area_ratio:.2f}  '
                    f'front={self._front_dist:.2f}m')

            # Grab if close enough (size-based)
            if area_ratio >= self.GRAB_AREA_RATIO:
                self._halt()
                self.get_logger().info(f'Close enough (ratio={area_ratio:.2f}). Grabbing.')
                self._state = 'GRAB'
                self._state_start = now
                return

            # Centering while approaching (milder correction)
            pixel_err = detection['cx'] - self.IMG_WIDTH / 2.0
            angular = -self.APPROACH_ANGULAR_KP * pixel_err
            angular = max(0.03, min(0.15, abs(angular)))
            angular = math.copysign(angular, -pixel_err)

            self._move(self.APPROACH_SPEED, angular)
            return

        # ═══════════════════════════════════════════════════════════════════════
        # GRAB – lower servo, hold, then finish
        # ═══════════════════════════════════════════════════════════════════════
        if self._state == 'GRAB':
            self._halt()
            self._servo_lower()
            
            # Wait for servo to actuate then transition
            if now - self._state_start >= self.SERVO_GRAB_HOLD:
                self.get_logger().info('=== Sequence complete ===')
                self._servo_lift()
                self._state = 'WAITING'
                self._initial_bbox_area = 0.0
            return

        # Fallback
        self._halt()

    # ═══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════════════

    def destroy_node(self):
        self._halt()
        self._servo_lift()
        time.sleep(0.3)
        if self._servo_pwm:
            self._servo_pwm.stop()
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedCubeApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[STOP] User interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()