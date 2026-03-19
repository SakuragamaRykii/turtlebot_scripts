
#!/usr/bin/env python3
import math
import signal
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan


class RedCubeFinder(Node):
    def __init__(self):
        super().__init__('red_cube_finder_v6')

        # =========================
        # 参数：真机优先“稳”
        # =========================
        self.CONTROL_DT = 0.05
        self.STATUS_DT = 1.5

        # 速度
        self.TURN_MAX_SPEED = 0.38
        self.TURN_MIN_SPEED = 0.10
        self.TURN_KP = 1.4

        self.PATROL_SPEED = 0.045
        self.TRACK_SPEED = 0.040
        self.BACKOFF_SPEED = -0.040

        # 小步动作参数（停一下再看）
        self.SEARCH_STEP_DEG = 8.0
        self.PATROL_TURN_DEG = 90.0
        self.TRACK_RECOVER_DEG = 10.0
        self.BACKOFF_DIST = 0.05
        self.PATROL_STEP_DIST = 0.10
        self.TRACK_STEP_DIST = 0.05

        self.SETTLE_TIME = 0.18
        self.TURN_TOL_DEG = 2.0

        # 雷达安全距离
        self.ROTATE_CLEARANCE_DIST = 0.16
        self.EMERGENCY_FRONT_DIST = 0.12
        self.TARGET_STOP_FRONT_DIST = 0.18
        self.PATROL_TURN_FRONT_DIST = 0.28
        self.REAR_SAFE_DIST = 0.18

        # 目标检测阈值
        self.RED_MIN_BOX_AREA = 850
        self.RED_CONFIRM_FRAMES = 2
        self.RED_LOST_TIMEOUT = 0.35
        self.TRACK_ALIGN_PX = 34
        self.TRACK_ROTATE_ONLY_PX = 85
        self.TRACK_ABORT_MOVE_PX = 110
        self.TARGET_STOP_BOX_AREA = 24000

        self.MAX_TRACK_RECOVER_STEPS = 8
        self.SEARCH_FULL_TURN_MARGIN_DEG = 10.0

        # 相机视场角近似（用于像素误差 -> 转角）
        self.CAMERA_HFOV_DEG = 62.0

        # =========================
        # ROS2 通信
        # =========================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(self.CONTROL_DT, self.control_loop)
        self.status_timer = self.create_timer(self.STATUS_DT, self.status_loop)

        # =========================
        # 传感器数据
        # =========================
        self.has_scan = False
        self.has_odom = False
        self.has_image = False

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')
        self.rear_dist = float('inf')
        self.min_dist_all = float('inf')

        self.image_width = None
        self.target_visible = False
        self.target_cx = None
        self.target_box_area = 0.0
        self.target_bbox = None
        self.target_seen_frames = 0
        self.target_last_time = 0.0
        self.last_target_error_px = 0.0
        self.last_target_side = 1

        # 可选：场地方位图，仅做辅助显示，不参与主控制
        self.marker_name = None
        self._aruco_ready = False
        self._aruco_counter = 0
        self._setup_aruco()

        # =========================
        # 状态机
        # =========================
        self.state = 'WAIT_FOR_DATA'
        self.prev_state = None
        self.state_enter_time = time.monotonic()

        self.search_turned_rad = 0.0
        self.track_recover_steps = 0

        # 通用动作执行器
        self.action_active = False
        self.action_kind = None          # 'turn' / 'move'
        self.action_label = ''
        self.action_next_state = None

        self.turn_goal_yaw = 0.0
        self.turn_tol = math.radians(self.TURN_TOL_DEG)
        self.turn_count_to_search = False
        self.turn_start_yaw = 0.0

        self.move_start_x = 0.0
        self.move_start_y = 0.0
        self.move_goal_dist = 0.0
        self.move_speed = 0.0
        self.move_stop_front = None

        self.settle_until = 0.0
        self.settle_next_state = None

        self.completed = False
        self.stop_sent_once = False
        self.wait_signature = None

        print('[BOOT] Red cube finder started.')

    # =========================
    # 初始化可选方位图检测
    # =========================
    def _setup_aruco(self):
        try:
            if not hasattr(cv2, 'aruco'):
                return
            if not hasattr(cv2.aruco, 'getPredefinedDictionary'):
                return
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            if hasattr(cv2.aruco, 'ArucoDetector'):
                params = cv2.aruco.DetectorParameters()
                self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, params)
                self._aruco_mode = 'new'
            else:
                self._aruco_params = cv2.aruco.DetectorParameters_create()
                self._aruco_mode = 'old'
            self._aruco_ready = True
        except Exception:
            self._aruco_ready = False

    # =========================
    # 工具函数
    # =========================
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp_abs(self, value, min_abs, max_abs):
        if abs(value) < 1e-9:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        mag = min(max(abs(value), min_abs), max_abs)
        return sign * mag

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop_once(self):
        try:
            self.publish_cmd(0.0, 0.0)
        except Exception:
            pass

    def reliable_stop(self, repeat=20, delay=0.02):
        for _ in range(repeat):
            try:
                self.publish_cmd(0.0, 0.0)
            except Exception:
                pass
            time.sleep(delay)

    def ready(self):
        return self.has_scan and self.has_odom and self.has_image

    def state_age(self):
        return time.monotonic() - self.state_enter_time

    def current_target_error_px(self):
        if not self.target_visible or self.image_width is None or self.target_cx is None:
            return None
        return float(self.target_cx - self.image_width / 2.0)

    def target_recent(self):
        return (time.monotonic() - self.target_last_time) <= self.RED_LOST_TIMEOUT

    def target_confirmed(self):
        return (
            self.target_visible
            and self.target_box_area >= self.RED_MIN_BOX_AREA
            and self.target_seen_frames >= self.RED_CONFIRM_FRAMES
            and self.target_recent()
        )

    def traveled_distance(self):
        return math.hypot(self.x - self.move_start_x, self.y - self.move_start_y)

    def log_state(self, msg):
        print(f'[STATE] {msg}')

    def set_state(self, new_state, reason=''):
        if self.state == new_state:
            return
        self.prev_state = self.state
        self.state = new_state
        self.state_enter_time = time.monotonic()
        if reason:
            self.log_state(f'{self.prev_state} -> {new_state} | {reason}')
        else:
            self.log_state(f'{self.prev_state} -> {new_state}')

    def enter_settle(self, next_state, reason=''):
        self.action_active = False
        self.action_kind = None
        self.action_label = ''
        self.settle_next_state = next_state
        self.settle_until = time.monotonic() + self.SETTLE_TIME
        self.set_state('SETTLE', reason)

    def start_turn_delta(self, delta_rad, next_state, label, count_to_search=False):
        self.action_active = True
        self.action_kind = 'turn'
        self.action_label = label
        self.action_next_state = next_state
        self.turn_goal_yaw = self.normalize_angle(self.yaw + delta_rad)
        self.turn_tol = math.radians(self.TURN_TOL_DEG)
        self.turn_count_to_search = count_to_search
        self.turn_start_yaw = self.yaw

    def start_move(self, distance_m, speed, next_state, label, stop_front=None):
        self.action_active = True
        self.action_kind = 'move'
        self.action_label = label
        self.action_next_state = next_state
        self.move_start_x = self.x
        self.move_start_y = self.y
        self.move_goal_dist = abs(distance_m)
        self.move_speed = float(speed)
        self.move_stop_front = stop_front

    def abort_action_and_hold(self, next_state, reason):
        self.stop_once()
        self.enter_settle(next_state, reason)

    def complete_mission(self, reason):
        if self.completed:
            return
        self.completed = True
        self.action_active = False
        self.stop_once()
        self.set_state('COMPLETE', reason)
        print('[EVENT] Mission complete. Robot stopped in front of target.')

    def px_error_to_rad(self, px_error, scale=1.0, max_deg=14.0, min_deg=3.0):
        if self.image_width is None:
            return 0.0
        ratio = px_error / float(self.image_width)
        delta_deg = ratio * self.CAMERA_HFOV_DEG * scale
        if abs(delta_deg) < min_deg:
            delta_deg = min_deg if delta_deg >= 0.0 else -min_deg
        delta_deg = max(-max_deg, min(max_deg, delta_deg))
        return math.radians(delta_deg)

    # =========================
    # 回调函数
    # =========================
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = np.nan
        ranges[ranges < 0.05] = np.nan

        def sector_min(start_deg, end_deg):
            if start_deg <= end_deg:
                vals = ranges[start_deg:end_deg + 1]
            else:
                vals = np.concatenate((ranges[start_deg:360], ranges[0:end_deg + 1]))
            if np.all(np.isnan(vals)):
                return float('inf')
            return float(np.nanmin(vals))

        self.front_dist = sector_min(350, 10)
        self.left_dist = sector_min(60, 120)
        self.right_dist = sector_min(240, 300)
        self.rear_dist = sector_min(170, 190)
        self.min_dist_all = float(np.nanmin(ranges)) if not np.all(np.isnan(ranges)) else float('inf')
        self.has_scan = True

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)
        self.has_odom = True

    def image_callback(self, msg):
        self.has_image = True

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception:
            return

        self.image_width = frame.shape[1]

        # 可选：识别场地方位图，仅作调试输出
        self._aruco_counter += 1
        if self._aruco_ready and self._aruco_counter % 8 == 0:
            self.detect_wall_marker(frame)

        # 红色目标检测：优先“稳”
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 70, 40], dtype=np.uint8)
        upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([168, 70, 40], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 用较强的闭运算尽量填平方块上的孔洞
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k2, iterations=1)
        mask = cv2.dilate(mask, k2, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_score = 0

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box_area = w * h
            if box_area < self.RED_MIN_BOX_AREA:
                continue

            # 过滤非常扁的误检
            aspect = w / float(max(h, 1))
            if aspect < 0.35 or aspect > 2.8:
                continue

            # 稍微偏向选择“更大、更靠中间”的目标
            cx = x + w / 2.0
            center_bonus = 1.0 - min(abs(cx - self.image_width / 2.0) / (self.image_width / 2.0), 1.0)
            score = box_area * (0.85 + 0.15 * center_bonus)

            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        if best_box is not None:
            x, y, w, h = best_box
            self.target_visible = True
            self.target_bbox = best_box
            self.target_box_area = float(w * h)
            self.target_cx = x + w / 2.0
            self.target_last_time = time.monotonic()
            self.target_seen_frames += 1

            err = self.current_target_error_px()
            if err is not None:
                self.last_target_error_px = err
                self.last_target_side = -1 if err < 0.0 else 1
        else:
            self.target_visible = False
            self.target_bbox = None
            self.target_box_area = 0.0
            self.target_cx = None
            self.target_seen_frames = 0

    def detect_wall_marker(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._aruco_mode == 'new':
                corners, ids, _ = self._aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)

            if ids is None or len(ids) == 0:
                return

            marker_map = {7: 'North', 23: 'East', 42: 'South', 0: 'West'}
            names = []
            for mid in ids.flatten().tolist():
                if mid in marker_map:
                    names.append(marker_map[mid])
            if names:
                self.marker_name = names[0]
        except Exception:
            pass

    # =========================
    # 主控制逻辑
    # =========================
    def control_loop(self):
        # 完成后持续保停
        if self.completed:
            self.stop_once()
            return

        # 等待数据
        if not self.ready():
            sig = (self.has_scan, self.has_odom, self.has_image)
            if sig != self.wait_signature:
                self.wait_signature = sig
                print(f'[WAIT] scan={self.has_scan} odom={self.has_odom} image={self.has_image}')
            self.stop_once()
            return

        # 刚刚准备好，启动一次完整搜索
        if self.state == 'WAIT_FOR_DATA':
            self.search_turned_rad = 0.0
            self.track_recover_steps = 0
            self.set_state('SEARCH_OBSERVE', 'All sensors ready. Start full search.')
            return

        # SETTLE：动作之间必须停一下再做下一步
        if self.state == 'SETTLE':
            self.stop_once()
            if time.monotonic() >= self.settle_until:
                nxt = self.settle_next_state
                self.settle_next_state = None
                self.set_state(nxt, 'Settle finished.')
            return

        # 通用优先级 1：如果正在搜索/巡航/转向时突然看到目标，必须立刻停车抢占
        if self.action_active and self.action_kind in ('turn', 'move'):
            if self.action_label.startswith(('search', 'patrol', 'escape')) and self.target_confirmed():
                self.abort_action_and_hold('TRACK_OBSERVE', 'Target found. Stop current motion immediately.')
                return

        # 通用优先级 2：前方太近，必须停车
        if self.action_active and self.action_kind == 'move':
            if self.front_dist <= self.EMERGENCY_FRONT_DIST and self.move_speed > 0.0:
                self.abort_action_and_hold('PATROL_OBSERVE', 'Emergency stop. Obstacle too close in front.')
                return

        # 执行通用动作
        if self.action_active:
            if self.action_kind == 'turn':
                self.run_turn_action()
                return
            if self.action_kind == 'move':
                self.run_move_action()
                return

        # 没有动作时，进入状态逻辑
        if self.state == 'SEARCH_OBSERVE':
            self.handle_search_observe()
        elif self.state == 'PATROL_OBSERVE':
            self.handle_patrol_observe()
        elif self.state == 'TRACK_OBSERVE':
            self.handle_track_observe()
        elif self.state == 'COMPLETE':
            self.stop_once()
        else:
            self.stop_once()

    # =========================
    # 动作执行器
    # =========================
    def run_turn_action(self):
        error = self.normalize_angle(self.turn_goal_yaw - self.yaw)
        if abs(error) <= self.turn_tol:
            if self.turn_count_to_search:
                step_done = abs(self.normalize_angle(self.yaw - self.turn_start_yaw))
                self.search_turned_rad += step_done
            self.stop_once()
            self.enter_settle(self.action_next_state, f'Turn done: {self.action_label}')
            return

        ang = self.clamp_abs(self.TURN_KP * error, self.TURN_MIN_SPEED, self.TURN_MAX_SPEED)
        self.publish_cmd(0.0, ang)

    def run_move_action(self):
        dist_done = self.traveled_distance()

        # 跟踪前进时：目标一旦丢失或横向跑太快，先停再看
        if self.action_label == 'track_forward':
            if self.front_dist <= self.TARGET_STOP_FRONT_DIST:
                self.complete_mission('Reached target stop distance.')
                return
            if not self.target_recent():
                self.abort_action_and_hold('TRACK_OBSERVE', 'Target lost during forward step.')
                return
            err = self.current_target_error_px()
            if err is not None and abs(err) > self.TRACK_ABORT_MOVE_PX:
                self.abort_action_and_hold('TRACK_OBSERVE', 'Target moved sideways. Stop and re-align.')
                return

        if self.move_speed > 0.0 and self.move_stop_front is not None and self.front_dist <= self.move_stop_front:
            self.stop_once()
            self.enter_settle(self.action_next_state, f'Front threshold reached during {self.action_label}')
            return

        if self.move_speed < 0.0 and self.rear_dist <= self.REAR_SAFE_DIST:
            self.stop_once()
            self.enter_settle(self.action_next_state, f'Rear threshold reached during {self.action_label}')
            return

        if dist_done >= self.move_goal_dist:
            self.stop_once()
            self.enter_settle(self.action_next_state, f'Move done: {self.action_label}')
            return

        self.publish_cmd(self.move_speed, 0.0)

    # =========================
    # 状态逻辑
    # =========================
    def handle_search_observe(self):
        if self.target_confirmed():
            self.track_recover_steps = 0
            self.stop_once()
            self.enter_settle('TRACK_OBSERVE', 'Target confirmed during search.')
            return

        # 完整转了一圈，进入巡航
        margin = math.radians(self.SEARCH_FULL_TURN_MARGIN_DEG)
        if self.search_turned_rad >= (2.0 * math.pi - margin):
            self.search_turned_rad = 0.0
            self.set_state('PATROL_OBSERVE', 'Full search finished. Start patrol.')
            return

        # 如果离墙太近，不原地转，先脱离再说
        if self.min_dist_all <= self.ROTATE_CLEARANCE_DIST:
            if self.rear_dist > self.REAR_SAFE_DIST + 0.05:
                self.start_move(self.BACKOFF_DIST, self.BACKOFF_SPEED, 'SEARCH_OBSERVE', 'escape_backoff')
            else:
                side = 1.0 if self.left_dist >= self.right_dist else -1.0
                self.start_turn_delta(math.radians(20.0) * side, 'SEARCH_OBSERVE', 'escape_turn')
            return

        # 搜索采用“小角度转 -> 停 -> 看”
        self.start_turn_delta(
            math.radians(self.SEARCH_STEP_DEG),
            'SEARCH_OBSERVE',
            'search_turn',
            count_to_search=True,
        )

    def handle_patrol_observe(self):
        if self.target_confirmed():
            self.track_recover_steps = 0
            self.stop_once()
            self.enter_settle('TRACK_OBSERVE', 'Target confirmed during patrol.')
            return

        # 太靠近障碍，先脱离
        if self.min_dist_all <= self.ROTATE_CLEARANCE_DIST:
            if self.rear_dist > self.REAR_SAFE_DIST + 0.05:
                self.start_move(self.BACKOFF_DIST, self.BACKOFF_SPEED, 'PATROL_OBSERVE', 'escape_backoff')
            else:
                side = 1.0 if self.left_dist >= self.right_dist else -1.0
                self.start_turn_delta(math.radians(25.0) * side, 'PATROL_OBSERVE', 'escape_turn')
            return

        # 前方近，左转 90 度
        if self.front_dist <= self.PATROL_TURN_FRONT_DIST:
            self.start_turn_delta(math.radians(self.PATROL_TURN_DEG), 'PATROL_OBSERVE', 'patrol_turn')
            return

        # 否则前进一步
        self.start_move(
            self.PATROL_STEP_DIST,
            self.PATROL_SPEED,
            'PATROL_OBSERVE',
            'patrol_forward',
            stop_front=self.PATROL_TURN_FRONT_DIST,
        )

    def handle_track_observe(self):
        # 成功条件
        if self.front_dist <= self.TARGET_STOP_FRONT_DIST:
            self.complete_mission('Reached target stop distance.')
            return
        if self.target_box_area >= self.TARGET_STOP_BOX_AREA and self.target_recent():
            self.complete_mission('Target is large enough in image. Stop.')
            return

        # 目标还看得见
        if self.target_confirmed():
            self.track_recover_steps = 0
            err = self.current_target_error_px()
            if err is None:
                self.enter_settle('SEARCH_OBSERVE', 'Invalid target reading. Restart search.')
                return

            self.last_target_error_px = err
            self.last_target_side = -1 if err < 0.0 else 1

            # 偏差大：先转，不前进
            if abs(err) > self.TRACK_ROTATE_ONLY_PX:
                delta = self.px_error_to_rad(err, scale=1.00, max_deg=12.0, min_deg=4.0)
                self.start_turn_delta(delta, 'TRACK_OBSERVE', 'track_align_big')
                return

            # 偏差中等：小角度修正
            if abs(err) > self.TRACK_ALIGN_PX:
                delta = self.px_error_to_rad(err, scale=0.70, max_deg=8.0, min_deg=2.5)
                self.start_turn_delta(delta, 'TRACK_OBSERVE', 'track_align_small')
                return

            # 已经比较正，对前走一步
            self.start_move(
                self.TRACK_STEP_DIST,
                self.TRACK_SPEED,
                'TRACK_OBSERVE',
                'track_forward',
                stop_front=self.TARGET_STOP_FRONT_DIST,
            )
            return

        # 目标暂时看不见：做有限次小范围恢复，失败后重新完整搜索
        if self.track_recover_steps < self.MAX_TRACK_RECOVER_STEPS:
            self.track_recover_steps += 1
            delta = math.radians(self.TRACK_RECOVER_DEG) * self.last_target_side
            self.start_turn_delta(delta, 'TRACK_OBSERVE', 'track_recover')
            return

        self.search_turned_rad = 0.0
        self.track_recover_steps = 0
        self.enter_settle('SEARCH_OBSERVE', 'Target lost. Return to full search.')

    # =========================
    # 状态输出
    # =========================
    def status_loop(self):
        if not self.ready():
            return

        target_flag = 'yes' if self.target_confirmed() else 'no'
        marker_text = self.marker_name if self.marker_name else '-'
        print(
            f'[STATUS] state={self.state} '
            f'front={self.front_dist:.2f} '
            f'left={self.left_dist:.2f} '
            f'right={self.right_dist:.2f} '
            f'target={target_flag} '
            f'area={int(self.target_box_area)} '
            f'marker={marker_text}'
        )

    # =========================
    # 关闭
    # =========================
    def shutdown_safely(self, reason='Shutdown requested'):
        try:
            print(f'[EVENT] {reason}')
        except Exception:
            pass

        try:
            self.action_active = False
            self.completed = True
            self.reliable_stop(repeat=30, delay=0.02)
        except Exception:
            pass

        try:
            self.control_timer.cancel()
            self.status_timer.cancel()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = RedCubeFinder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_safely('Keyboard interrupt received. Robot stop requested.')
    finally:
        try:
            node.shutdown_safely('Final stop sequence.')
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
