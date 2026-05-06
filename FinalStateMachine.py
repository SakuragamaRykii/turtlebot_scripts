#!/usr/bin/env python3
"""
TurtleBot3 – Red Cube Detection, Approach & Pickup
----------------------------------------------------
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

    # ── Image parameters ─────────────────────────────────────────────────────
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    ROI_WIDTH = 200
    ROI_HEIGHT = 480

    # ── Rotation parameters ──────────────────────────────────────────────────
    SEARCH_ANGULAR_SPEED = 0.35      # rad/s during full rotation search
    CENTERING_ANGULAR_SPEED_MIN = 0.08
    CENTERING_ANGULAR_SPEED_MAX = 0.30
    CENTERING_KP = 0.004             # proportional gain for visual centering

    # ── Approach parameters ──────────────────────────────────────────────────
    APPROACH_LINEAR_SPEED = 0.10     # m/s forward
    GRAB_DISTANCE_THRESHOLD = 0.15   # m (estimated) – trigger grab
    SIZE_RATIO_GRAB = 2.5            # if current bbox area >= initial * ratio → close

    # ── Safety distances ─────────────────────────────────────────────────────
    MIN_FRONT_DIST = 0.18            # m – stop if LiDAR front is closer
    EMERGENCY_DIST = 0.12            # m – immediate halt any direction

    # ── Timing ───────────────────────────────────────────────────────────────
    CONTROL_DT = 0.05                # s – control loop period

    # ── Detection parameters ─────────────────────────────────────────────────
    MIN_CONTOUR_AREA = 200
    MIN_BBOX_WIDTH = 12
    MIN_BBOX_HEIGHT = 12
    MIN_SOLIDITY = 0.75
    MIN_FILL_RATIO = 0.25
    COLOR_CONFIDENCE_THRESHOLD = 0.65

    # ── Servo ────────────────────────────────────────────────────────────────
    SERVO_PIN = 12
    SERVO_UP_DUTY = 5.5
    SERVO_DOWN_DUTY = 9.0
    SERVO_FREQ = 50

    def __init__(self):
        super().__init__('red_cube_approach')

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            LaserScan, '/scan', self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self._image_callback,
            qos_profile_sensor_data)

        # ── Servo ────────────────────────────────────────────────────────────
        self._servo = None
        if HAS_GPIO:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.SERVO_PIN, GPIO.OUT)
                self._servo = GPIO.PWM(self.SERVO_PIN, self.SERVO_FREQ)
                self._servo.start(self.SERVO_UP_DUTY)
                self.get_logger().info(f'Servo initialized on GPIO {self.SERVO_PIN} – UP position')
            except Exception as exc:
                self.get_logger().error(f'Servo init failed: {exc}')
        else:
            self.get_logger().info('Servo control simulated')

        # ── Sensor state ─────────────────────────────────────────────────────
        self._front_distance = float('inf')
        self._left_distance = float('inf')
        self._right_distance = float('inf')
        self._back_distance = float('inf')
        self._has_scan = False

        # ── Vision state ─────────────────────────────────────────────────────
        self._has_image = False
        self._current_frame = None
        self._red_detected = False
        self._detection = None       # dict with bbox, center, area
        self._initial_bbox_area = None
        self._frame_lock = threading.Lock()

        # ── State machine ────────────────────────────────────────────────────
        self._state = 'INIT'
        self._state_start_time = time.monotonic()
        self._full_rotation_start_yaw = None
        self._full_rotation_completed = False
        self._move_command = Twist()

        # ── Image processing kernel ──────────────────────────────────────────
        self._morph_kernel = np.ones((5, 5), np.uint8)

        # ── Control timer ────────────────────────────────────────────────────
        self.create_timer(self.CONTROL_DT, self._control_loop)

        self.get_logger().info('Red Cube Approach Node ready. Press Enter to start.')

        # Console thread for start command
        self._console_thread = threading.Thread(target=self._console_listener, daemon=True)
        self._console_thread.start()

    # ═══════════════════════════════════════════════════════════════════════════
    # Servo helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _servo_up(self):
        """Raise the servo (release position)."""
        self.get_logger().info('Servo → UP')
        if self._servo is not None:
            self._servo.ChangeDutyCycle(self.SERVO_UP_DUTY)

    def _servo_down(self):
        """Lower the servo (grab position)."""
        self.get_logger().info('Servo → DOWN (GRAB)')
        if self._servo is not None:
            self._servo.ChangeDutyCycle(self.SERVO_DOWN_DUTY)

    # ═══════════════════════════════════════════════════════════════════════════
    # Sensor callbacks
    # ═══════════════════════════════════════════════════════════════════════════

    def _scan_callback(self, msg: LaserScan):
        """Process LiDAR data – smooth distances for four cardinal directions."""
        ranges = msg.ranges
        n = len(ranges)

        def safe_min(indices):
            vals = [ranges[i] for i in indices
                    if 0 <= i < n and math.isfinite(ranges[i]) and ranges[i] > 0.04]
            return min(vals) if vals else float('inf')

        front = list(range(0, 16)) + list(range(n - 16, n))
        back = list(range(n // 2 - 16, n // 2 + 16))
        left = list(range(75, 105))
        right = list(range(255, 285))

        raw_front = safe_min(front)
        raw_back = safe_min(back)
        raw_left = safe_min(left)
        raw_right = safe_min(right)

        alpha = 0.45
        if not self._has_scan:
            self._front_distance = raw_front
            self._back_distance = raw_back
            self._left_distance = raw_left
            self._right_distance = raw_right
            self._has_scan = True
        else:
            self._front_distance = alpha * self._front_distance + (1 - alpha) * raw_front
            self._back_distance = alpha * self._back_distance + (1 - alpha) * raw_back
            self._left_distance = alpha * self._left_distance + (1 - alpha) * raw_left
            self._right_distance = alpha * self._right_distance + (1 - alpha) * raw_right

    def _image_callback(self, msg: CompressedImage):
        """Decode compressed image and run red cube detection."""
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception:
            return

        detection = self._detect_red_cube(frame)

        with self._frame_lock:
            self._current_frame = frame
            self._has_image = True
            if detection is not None:
                self._red_detected = True
                self._detection = detection
            else:
                self._red_detected = False
                self._detection = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Red cube detection
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_red_cube(self, frame):
        """
        Find the most prominent red cube in the frame.
        Returns dict with 'cx', 'cy', 'bbox_width', 'bbox_height', 'area',
        or None if no valid cube is found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around 0°/180° in HSV
        mask1 = cv2.inRange(hsv, np.array([0, 90, 40]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([168, 90, 40]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # BGR verification – red channel dominates
        b, g, r = cv2.split(frame)
        bgr_mask = np.zeros(red_mask.shape, dtype=np.uint8)
        bgr_mask[(r >= 65) & (r > g + 20) & (r > b + 20)] = 255
        combined = cv2.bitwise_and(red_mask, bgr_mask)

        # Morphological cleanup
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self._morph_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._morph_kernel)
        dilated = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CONTOUR_AREA:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.MIN_BBOX_WIDTH or bh < self.MIN_BBOX_HEIGHT:
                continue

            # Aspect ratio filter (cubes are roughly square)
            aspect = bw / float(bh)
            if not (0.55 <= aspect <= 1.85):
                continue

            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity < self.MIN_SOLIDITY:
                continue

            # Fill ratio within bounding box
            roi_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted = cnt - np.array([[x, y]])
            cv2.drawContours(roi_mask, [shifted], -1, 255, -1)
            fill = np.count_nonzero(roi_mask) / float(bw * bh)
            if fill < self.MIN_FILL_RATIO:
                continue

            # Color confidence inside the contour
            roi_hsv = hsv[y:y + bh, x:x + bw]
            roi_mask_bool = roi_mask > 0
            hue_pixels = roi_hsv[:, :, 0][roi_mask_bool]
            sat_pixels = roi_hsv[:, :, 1][roi_mask_bool]

            if len(hue_pixels) == 0:
                continue

            red_hue_count = np.sum((hue_pixels <= 12) | (hue_pixels >= 168))
            high_sat_count = np.sum(sat_pixels >= 110)
            confidence = (red_hue_count / len(hue_pixels)) * (high_sat_count / len(sat_pixels))
            if confidence < self.COLOR_CONFIDENCE_THRESHOLD:
                continue

            if area > best_area:
                best_area = area
                best_candidate = {
                    'cx': float(x + bw / 2.0),
                    'cy': float(y + bh / 2.0),
                    'bbox_width': bw,
                    'bbox_height': bh,
                    'area': float(area),
                    'contour': cnt,
                    'bbox_x': x,
                    'bbox_y': y,
                }

        return best_candidate

    # ═══════════════════════════════════════════════════════════════════════════
    # ROI check – is the cube within the 200×480 center region?
    # ═══════════════════════════════════════════════════════════════════════════

    def _is_in_roi(self, detection) -> bool:
        """
        Check if the detection's center lies within the central 200×480 region
        of the 640×480 image.
        """
        roi_left = (self.IMG_WIDTH - self.ROI_WIDTH) // 2
        roi_right = roi_left + self.ROI_WIDTH
        cx = detection['cx']
        return roi_left <= cx <= roi_right

    # ═══════════════════════════════════════════════════════════════════════════
    # Distance estimation via relative size
    # ═══════════════════════════════════════════════════════════════════════════

    def _estimate_distance_from_size(self, current_area: float) -> float:
        """
        Estimate distance based on the ratio of initial to current bounding-box area.
        distance ≈ initial_distance * sqrt(initial_area / current_area)
        We use a nominal initial distance of 2.0 m when the cube is first locked.
        """
        if self._initial_bbox_area is None or current_area <= 0:
            return float('inf')

        # Assumption: the cube was ~2.0 m away when first detected
        nominal_initial_distance = 2.0
        ratio = self._initial_bbox_area / current_area
        estimated = nominal_initial_distance * math.sqrt(ratio)
        return max(0.05, estimated)  # clamp to realistic minimum

    # ═══════════════════════════════════════════════════════════════════════════
    # Motion commands
    # ═══════════════════════════════════════════════════════════════════════════

    def _publish_velocity(self, linear: float = 0.0, angular: float = 0.0):
        """Publish a Twist message with the given linear (m/s) and angular (rad/s)."""
        self._move_command.linear.x = float(linear)
        self._move_command.angular.z = float(angular)
        self.cmd_pub.publish(self._move_command)

    def _stop(self):
        """Stop all motion."""
        self._publish_velocity(0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Safety checks
    # ═══════════════════════════════════════════════════════════════════════════

    def _safety_ok(self) -> bool:
        """Return True if it is safe to move forward."""
        if not self._has_scan:
            return True  # no data yet, allow motion cautiously
        return (self._front_distance > self.EMERGENCY_DIST and
                self._left_distance > self.EMERGENCY_DIST and
                self._right_distance > self.EMERGENCY_DIST and
                self._back_distance > self.EMERGENCY_DIST)

    def _should_stop_approach(self) -> bool:
        """Return True if the robot is too close to an obstacle to keep approaching."""
        if not self._has_scan:
            return False
        return self._front_distance <= self.MIN_FRONT_DIST

    # ═══════════════════════════════════════════════════════════════════════════
    # Console listener
    # ═══════════════════════════════════════════════════════════════════════════

    def _console_listener(self):
        """Listen for Enter key to start the sequence."""
        while True:
            try:
                input()
                self.get_logger().info('Start command received.')
                self._state = 'SEARCH_ROTATE'
                self._state_start_time = time.monotonic()
                self._initial_bbox_area = None
                self._full_rotation_start_yaw = None
                self._full_rotation_completed = False
            except EOFError:
                break

    # ═══════════════════════════════════════════════════════════════════════════
    # Main control loop
    # ═══════════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        """State machine executed at CONTROL_DT Hz."""

        # ── Emergency stop override ───────────────────────────────────────────
        if not self._safety_ok():
            self._stop()
            self.get_logger().warn('Emergency stop – obstacle too close')
            self._state = 'INIT'
            return

        # ── INIT state – wait for command ─────────────────────────────────────
        if self._state == 'INIT':
            self._stop()
            return

        # ── SEARCH_ROTATE – perform a full rotation looking for red cube ──────
        if self._state == 'SEARCH_ROTATE':
            # Rotate at constant speed
            self._publish_velocity(0.0, self.SEARCH_ANGULAR_SPEED)

            with self._frame_lock:
                detected = self._red_detected

            if detected:
                # Found a cube – lock initial size and switch to centering
                self._stop()
                with self._frame_lock:
                    self._initial_bbox_area = self._detection['area']
                self.get_logger().info(
                    f'Red cube found. Initial area = {self._initial_bbox_area:.0f} px²')
                self._state = 'CENTER_IN_ROI'
                self._state_start_time = time.monotonic()
                return

            # Check if a full rotation has completed (approximate by time)
            elapsed = time.monotonic() - self._state_start_time
            full_rotation_time = (2 * math.pi) / self.SEARCH_ANGULAR_SPEED
            if elapsed > full_rotation_time + 0.5:
                self._stop()
                self.get_logger().info('Full rotation complete – no red cube found. Returning to INIT.')
                self._state = 'INIT'
            return

        # ── CENTER_IN_ROI – rotate until cube is inside the 200×480 crop ─────
        if self._state == 'CENTER_IN_ROI':
            with self._frame_lock:
                detection = self._detection
                detected = self._red_detected

            # Lost the cube – go back to full rotation search
            if not detected or detection is None:
                self._stop()
                self.get_logger().info('Cube lost during centering – restarting full rotation.')
                self._state = 'SEARCH_ROTATE'
                self._state_start_time = time.monotonic()
                self._initial_bbox_area = None
                return

            if self._is_in_roi(detection):
                # Cube is centered – begin approach
                self._stop()
                self.get_logger().info('Cube in ROI. Starting approach.')
                self._state = 'APPROACH'
                self._state_start_time = time.monotonic()
                return

            # Cube visible but outside ROI – rotate to center it
            cx = detection['cx']
            roi_center_x = self.IMG_WIDTH / 2.0
            error_px = cx - roi_center_x
            error_normalized = error_px / (self.IMG_WIDTH / 2.0)

            # Proportional control – negative so robot turns toward the cube
            angular = -self.CENTERING_KP * error_px
            # Clamp to min/max
            angular = max(self.CENTERING_ANGULAR_SPEED_MIN,
                          min(self.CENTERING_ANGULAR_SPEED_MAX, abs(angular)))
            angular = math.copysign(angular, -error_px)

            self._publish_velocity(0.0, angular)
            return

        # ── APPROACH – drive forward while monitoring relative size ───────────
        if self._state == 'APPROACH':
            with self._frame_lock:
                detection = self._detection
                detected = self._red_detected

            # Safety stop from LiDAR
            if self._should_stop_approach():
                self._stop()
                self.get_logger().info('Approach halted by LiDAR – grabbing now.')
                self._state = 'GRAB'
                return

            # Lost the cube – assume it is under the robot, grab
            if not detected or detection is None:
                self._stop()
                self.get_logger().info('Cube lost (likely underneath) – grabbing.')
                self._state = 'GRAB'
                return

            # Estimate distance from size ratio
            current_area = detection['area']
            distance_estimate = self._estimate_distance_from_size(current_area)
            size_ratio = current_area / max(self._initial_bbox_area, 1.0)

            self.get_logger().info(
                f'Approaching – area={current_area:.0f} ratio={size_ratio:.2f} '
                f'est_distance={distance_estimate:.2f} m')

            # Check if close enough to grab
            if distance_estimate <= self.GRAB_DISTANCE_THRESHOLD or size_ratio >= self.SIZE_RATIO_GRAB:
                self._stop()
                self.get_logger().info('Cube close enough – grabbing.')
                self._state = 'GRAB'
                return

            # Keep cube centered while approaching
            cx = detection['cx']
            roi_center_x = self.IMG_WIDTH / 2.0
            error_px = cx - roi_center_x

            angular = -self.CENTERING_KP * error_px * 0.5  # milder correction while moving
            angular = max(0.03, min(0.15, abs(angular)))
            angular = math.copysign(angular, -error_px)

            self._publish_velocity(self.APPROACH_LINEAR_SPEED, angular)
            return

        # ── GRAB – lower servo, then finish ───────────────────────────────────
        if self._state == 'GRAB':
            self._stop()
            self._servo_down()
            time.sleep(0.6)
            self.get_logger().info('Cube grabbed. Sequence complete. Returning to INIT.')
            self._state = 'INIT'
            self._initial_bbox_area = None
            return

        # Fallback
        self._stop()

    # ═══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════════════════════

    def destroy_node(self):
        self._stop()
        self._servo_up()
        if self._servo is not None:
            self._servo.stop()
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RedCubeApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('[STOP] Interrupted by user.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()