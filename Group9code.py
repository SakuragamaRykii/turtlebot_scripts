"""
================================================================================
  ROBOT STATE MACHINE  —  V9 (The Ultimate God-Tier Edition)
  六大神级外挂已全面实装，专治各种实车物理不服：
  1. 护臂刹车 (WALL_STOP_DIST = 0.48): 保护32cm尺子，绝不撞墙。
  2. 极速起步 (PUSH_BLIND_TIME = 0.5): 缩短盲推防撞死，仅用0.5秒防假摔。
  3. 像素回中法 (Pixel Balancing): 抛弃盲开，看任意墙精准退回场中心。
  4. 智能垃圾过滤: 在红区无视红块，在蓝区无视蓝块，绝不重复搬砖。
  5. 🌟 [V9新] 斜眼雷达 (Oblique Scan): 测距范围改为20~45度，完美绕开爪子里的方块！
  6. 🌟 [V9新] 像素物理缓冲 (Zone Buffer): 利用 a_error > 100 完美解决中线方块压线误判！
================================================================================
"""

import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String, Int32
from rclpy.qos import qos_profile_sensor_data

POINTBLANK_Y = 228.0      

# 🚨 根据你的最新赛规设定：左边0是红区，右边23是蓝区
COLOR_TO_WALL = {"RED": 0, "BLUE": 23}

SCAN_SPEED = 0.22
CENTRE_SPEED = 0.18
APPROACH_FWD = 0.07
APPROACH_P_GAIN = 0.003
PUSH_FWD = 0.15          
PUSH_P_GAIN = 0.002
PIVOT_SPEED = 0.22

# 🌟 V9 外挂 1：护臂防撞参数
WALL_STOP_DIST = 0.52     # 32cm机械臂 + 缓冲 = 0.48米 (雷达现在看斜前方)
REVERSE_FWD = -0.20       # 倒车速度
REVERSE_DURATION = 1.8    # 吐出方块的倒车时间

# 🌟 V9 外挂 3：像素回中参数
CENTER_TARGET_WIDTH = 41.0 # 场中心的二维码基准宽度
WIDTH_TOLERANCE = 2.0      # 宽度允许误差 (39.0 ~ 43.0 都算在中心)

CENTRE_PIXEL_THRESH = 20.0
PIVOT_PIXEL_THRESH = 20.0
APPROACH_LOST_LIMIT = 8
PUSH_TIMEOUT = 12.0
PIVOT_TIMEOUT = 35.0

# 🌟 V9 外挂 2：极速起步防假摔 (改短到 0.5 秒，防止闭眼撞墙)
PUSH_BLIND_TIME = 0.5    

class StateMachineV9(Node):
    def __init__(self):
        super().__init__('state_machine_v9')

        self.create_subscription(Bool, '/target_found', self._cb_t_found, 10)
        self.create_subscription(Float32, '/color_error', self._cb_t_error, 10)
        self.create_subscription(Float32, '/target_y', self._cb_t_y, 10)
        self.create_subscription(String, '/target_color', self._cb_t_color, 10)
        self.create_subscription(Bool, '/aruco_detected', self._cb_a_found, 10)
        self.create_subscription(Int32, '/aruco_id', self._cb_a_id, 10)
        self.create_subscription(Float32, '/aruco_error', self._cb_a_error, 10)
        self.create_subscription(Float32, '/aruco_width', self._cb_a_width, 10)
        self.create_subscription(LaserScan, '/scan', self._cb_scan, qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.t_found = False
        self.t_error = 0.0
        self.t_y = 0.0
        self.t_color = "NONE"
        self.a_found = False
        self.a_id = -1
        self.a_error = 0.0
        self.a_width = 0.0
        self.forward_dist = 999.0

        self.state = "SCAN"
        self.carried_color = "NONE"
        self.target_wall_id = -1
        self.lost_count = 0
        self.reversing = False
        self.reverse_start = 0.0
        self.push_start = 0.0
        self.pivot_start = 0.0
        self.deliveries = 0
       
        self.current_zone = "UNKNOWN"
        self.confidence_counter = 0
       
        self.last_y = 0.0              
        self.last_color = "NONE"      
        self.last_seen_a_id = -1      

        self.timer = self.create_timer(0.1, self._loop)
        self.get_logger().info("========================================")
        self.get_logger().info("[SYSTEM] 🚀 V9 Brain Started! (Oblique Lidar & Zone Buffer Active!)")
        self.get_logger().info("========================================")

    def _cb_t_found(self, msg): self.t_found = msg.data
    def _cb_t_error(self, msg): self.t_error = msg.data
    def _cb_t_y(self, msg): self.t_y = msg.data
    def _cb_t_color(self, msg): self.t_color = msg.data
    def _cb_a_found(self, msg): self.a_found = msg.data
    def _cb_a_id(self, msg): self.a_id = msg.data
    def _cb_a_error(self, msg): self.a_error = msg.data
    def _cb_a_width(self, msg): self.a_width = msg.data

    # ==========================================================
    # 🌟 V9 外挂 5：斜眼看路法 (抛弃正前方，专看左右斜前方)
    # ==========================================================
    def _cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0: return
       
        inc = msg.angle_increment if msg.angle_increment > 0 else 0.0175
       
        # 我们查 20度 到 45度 的区间，完美绕过中间的机械臂和方块
        idx_20 = int(math.radians(20) / inc)
        idx_45 = int(math.radians(45) / inc)
       
        valid_ranges = []
       
        # 扫左斜前方
        for i in range(idx_20, idx_45):
            dist = msg.ranges[i % n]
            if math.isfinite(dist) and dist > 0.15: # >0.15 忽略车体边缘
                valid_ranges.append(dist)
               
        # 扫右斜前方
        for i in range(-idx_45, -idx_20):
            dist = msg.ranges[i % n]
            if math.isfinite(dist) and dist > 0.15:
                valid_ranges.append(dist)
               
        # 只要斜前方扫到墙，就取最近的那个距离作为刹车依据
        if valid_ranges:
            self.forward_dist = min(valid_ranges)
        else:
            self.forward_dist = 999.0

    def _stop(self): self.cmd_pub.publish(Twist())
    def _spin(self, speed):
        cmd = Twist()
        cmd.angular.z = float(speed)
        self.cmd_pub.publish(cmd)
    def _drive(self, linear, angular=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)
       
    def _reset_approach(self):
        self.lost_count = 0
        self.last_y = 0.0
        self.last_color = "NONE"

    def _go_scan(self):
        self._reset_approach()
        self.carried_color = "NONE"
        self.target_wall_id = -1
        self.confidence_counter = 0
        self.state = "SCAN"
        self._stop()
        self.get_logger().info("[STATE CHANGE] 🔄 Back to SCAN (Looking for NEW blocks...)")

    def _start_reverse(self):
        self.reverse_start = time.time()
        self.reversing = True
        self.get_logger().info(f"[STATE CHANGE] 🔙 REVERSE (Backing up to release block...)")

    def _loop(self):
        # ==========================================================
        # 🌟 V9 外挂 6：像素级物理缓冲半场划分法 (专治中线压线Bug)
        # ==========================================================
        if self.a_found:
            # -- 红区判定 --
            if self.a_id == 0:
                self.current_zone = "RED_ZONE"
            # 看到 7 号墙，必须等它跑到画面最右边 (a_error > 100)，才确认跨过中线！
            elif self.a_id == 7 and self.a_error > 100.0:
                self.current_zone = "RED_ZONE"

            # -- 蓝区判定 --
            elif self.a_id == 23:
                self.current_zone = "BLUE_ZONE"
            # 看到 42 号墙，必须等它跑到画面最右边 (a_error > 100)，才确认跨过中线！
            elif self.a_id == 42 and self.a_error > 100.0:
                self.current_zone = "BLUE_ZONE"
        # ==========================================================

        if self.reversing:
            if time.time() - self.reverse_start < REVERSE_DURATION:
                self._drive(REVERSE_FWD)
            else:
                self.reversing = False
                self._stop()
                self.state = "RETURN_CENTER"
                self.get_logger().info("[STATE CHANGE] 🎯 RETURN_CENTER (Using Pixel Balance to find center...)")
                self.last_seen_a_id = -1
            return

        if   self.state == "SCAN":          self._state_scan()
        elif self.state == "CENTRE":        self._state_centre()
        elif self.state == "APPROACH":      self._state_approach()
        elif self.state == "PIVOT":         self._state_pivot()
        elif self.state == "PUSH":          self._state_push()
        elif self.state == "RETURN_CENTER": self._state_return_center()

    def _state_return_center(self):
        if self.a_found:
            if getattr(self, 'last_seen_a_id', -1) != self.a_id:
                self.get_logger().info(f"👀 [PIXEL RETURN] 锁定回中灯塔，墙 ID: {self.a_id}，当前宽度: {self.a_width:.1f}")
                self.last_seen_a_id = self.a_id

            err = self.a_error
            steer = -(err / 160.0) * PIVOT_SPEED
            steer = max(-PIVOT_SPEED, min(PIVOT_SPEED, steer))

            if abs(self.a_width - CENTER_TARGET_WIDTH) <= WIDTH_TOLERANCE:
                if abs(err) < PIVOT_PIXEL_THRESH:
                    self.get_logger().info(f"✅ [RETURN_CENTER] 中线抵达！(宽度 {self.a_width:.1f})，完成配送！")
                    self.deliveries += 1
                    self._go_scan()
                else:
                    self._drive(0.0, steer)
            elif self.a_width < (CENTER_TARGET_WIDTH - WIDTH_TOLERANCE):
                self._drive(0.12, steer)
            elif self.a_width > (CENTER_TARGET_WIDTH + WIDTH_TOLERANCE):
                self._drive(-0.12, steer)
        else:
            self.last_seen_a_id = -1
            self._spin(PIVOT_SPEED)

    def _state_scan(self):
        if self.t_found and self.t_color in ("RED", "BLUE"):
            # 🌟 智能过滤：我在红区，看到红块，直接无视！蓝区同理！
            if (self.t_color == "RED" and self.current_zone == "RED_ZONE") or \
               (self.t_color == "BLUE" and self.current_zone == "BLUE_ZONE"):
                self._spin(SCAN_SPEED)
            else:
                self.confidence_counter += 1
                if self.confidence_counter > 5:
                    self.get_logger().info(f"[STATE CHANGE] 🎯 SCAN → CENTRE (Locked onto {self.t_color} block!)")
                    self._stop()
                    self.confidence_counter = 0
                    self.state = "CENTRE"
                else:
                    self._spin(0.1)
        else:
            self.confidence_counter = 0
            self._spin(SCAN_SPEED)

    def _state_centre(self):
        if not self.t_found or self.t_color not in ("RED", "BLUE"):
            self.get_logger().warn("⚠️ [CENTRE → SCAN] Block lost while centring. Aborting.")
            self._go_scan()
            return
           
        error = self.t_error
        if abs(error) < CENTRE_PIXEL_THRESH:
            self.get_logger().info(f"[STATE CHANGE] 🚀 CENTRE → APPROACH (Centred! Pushing gas pedal!)")
            self._reset_approach()
            self._stop()
            self.state = "APPROACH"
        else:
            steer = -float(error) * CENTRE_SPEED / 80.0
            steer = max(-CENTRE_SPEED, min(CENTRE_SPEED, steer))
            self._spin(steer)

    def _state_approach(self):
        if self.t_found and self.t_color in ("RED", "BLUE"):
            self.lost_count = 0
            self.last_y = self.t_y  
            self.last_color = self.t_color  
            steer = -(self.t_error * APPROACH_P_GAIN)
            self._drive(APPROACH_FWD, steer)
        else:
            self.lost_count += 1
            if self.lost_count >= APPROACH_LOST_LIMIT:
                if self.last_y > POINTBLANK_Y:
                    self._confirm_capture(f"Swallowed into blind spot! (Last Y: {self.last_y:.0f} > 228)")
                else:
                    self.get_logger().warn(f"⚠️ [APPROACH → SCAN] Target lost too early! Fake capture rejected.")
                    self._go_scan()
            else:
                self._drive(APPROACH_FWD)

    def _confirm_capture(self, reason: str):
        color = self.last_color
        if color not in COLOR_TO_WALL:
            self._go_scan()
            return
           
        self.carried_color = color
        self.target_wall_id = COLOR_TO_WALL[color]
        self._reset_approach()
        self.get_logger().info(f"[STATE CHANGE] 🔄 APPROACH → PIVOT ({reason}. Turning to find Wall ID: {self.target_wall_id})")
        self._stop()
        self.pivot_start = time.time()
        self.state = "PIVOT"

    def _state_pivot(self):
        elapsed = time.time() - self.pivot_start
        if elapsed > PIVOT_TIMEOUT:
            self.get_logger().error("❌ [PIVOT → REVERSE] Pivot Timeout (Can't find wall). Dropping block.")
            self._start_reverse()
            return
           
        if self.a_found:
            if getattr(self, 'last_seen_a_id', -1) != self.a_id:
                self.get_logger().info(f"👀 [VISION RADAR] 报告！眼前扫过 ArUco 码，ID: {self.a_id} !")
                self.last_seen_a_id = self.a_id
               
            if self.a_id == self.target_wall_id:
                err = self.a_error
                if abs(err) < PIVOT_PIXEL_THRESH:
                    self.get_logger().info(f"[STATE CHANGE] 🚚 PIVOT → PUSH (Wall locked! Pushing block to wall...)")
                    self._stop()
                    self.push_start = time.time()
                    self.state = "PUSH"
                else:
                    steer = -(err / 160.0) * PIVOT_SPEED
                    steer = max(-PIVOT_SPEED, min(PIVOT_SPEED, steer))
                    self._spin(steer)
            else:
                self._spin(PIVOT_SPEED)
        else:
            self.last_seen_a_id = -1
            self._spin(PIVOT_SPEED)

    def _state_push(self):
        elapsed = time.time() - self.push_start
       
        # 过了 0.5 秒盲推期，雷达完全睁开眼睛（并且现在是斜眼看路，不怕方块挡！）
        if elapsed > PUSH_BLIND_TIME:
            if self.forward_dist < WALL_STOP_DIST:
                self.get_logger().warn(f"🛑 [PUSH → REVERSE] Wall reached! (Oblique Lidar dist = {self.forward_dist:.3f}m). Stopping push.")
                self._stop()
                self._start_reverse()
                return
               
        if elapsed > PUSH_TIMEOUT:
            self.get_logger().error("❌ [PUSH → REVERSE] Push Timeout (Took too long). Retreating.")
            self._stop()
            self._start_reverse()
            return
           
        if self.a_found and self.a_id == self.target_wall_id:
            steer = -(self.a_error * PUSH_P_GAIN)
        else:
            steer = 0.0
        self._drive(PUSH_FWD, steer)

def main(args=None):
    rclpy.init(args=args)
    node = StateMachineV9()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String, Int32

from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
import math  # 🌟 新增：用于计算红蓝方块的距离

from flask import Flask, Response
import threading
import time


class ColorTracking(Node):

    def __init__(self):
        super().__init__('color_tracking')

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.target_pub = self.create_publisher(Bool, '/target_found', 10)
        self.error_pub = self.create_publisher(Float32, '/color_error', 10)
        self.area_pub = self.create_publisher(Float32, '/target_area', 10)
        self.y_pub = self.create_publisher(Float32, '/target_y', 10)
        self.color_pub = self.create_publisher(String, '/target_color', 10)

        self.green_pub = self.create_publisher(Bool, '/green_detected', 10)
        self.green_y_pub = self.create_publisher(Float32, '/green_y', 10)

        self.aruco_pub = self.create_publisher(Bool, '/aruco_detected', 10)
        self.aruco_id_pub = self.create_publisher(Int32, '/aruco_id', 10)
        self.aruco_error_pub = self.create_publisher(Float32, '/aruco_error', 10)
        self.aruco_width_pub = self.create_publisher(Float32, '/aruco_width', 10)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params
            )
            self.use_new_aruco_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_aruco_api = False

        self.app = Flask(__name__)
        self.frame = None
        self.frame_lock = threading.Lock()
        threading.Thread(target=self.run_flask, daemon=True).start()

        self.last_print_time = 0.0

        self.last_target = {
            "found": False,
            "color": "NONE",
            "error": 0.0,
            "area": 0.0,
            "y": 0.0
        }

        self.lost_count = 0
        self.MAX_LOST_FRAMES = 8

        self.get_logger().info("Integrated Color Tracking Node Started (Billiard Break Edition!)")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg)
                if msg.encoding == "nv21":
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV21)
                else:
                    return
            except Exception:
                return

        if frame is None:
            return

        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        frame = cv2.LUT(frame, table)

        height, width, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # RED
        lower_red1 = np.array([0, 140, 60])
        upper_red1 = np.array([8, 255, 255])

        lower_red2 = np.array([172, 140, 60])
        upper_red2 = np.array([180, 255, 255])

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red += cv2.inRange(hsv, lower_red2, upper_red2)

        # BLUE
        lower_blue = np.array([102, 145, 55])
        upper_blue = np.array([132, 255, 255])

        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # GREEN TAPE
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([80, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)

        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets = []

        # RED TARGETS
        for contour in contours_red:
            area = cv2.contourArea(contour)
            if area < 200: continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0: continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 2.2: continue
            roi = hsv[y:y+h, x:x+w]
            if roi.size == 0: continue
            mean_s = np.mean(roi[:, :, 1])
            mean_v = np.mean(roi[:, :, 2])
            if mean_s < 120 or mean_v < 50: continue

            cx = x + w // 2
            bottom_y = y + h
            targets.append({
                "color": "RED", "area": area, "cx": cx, "bottom_y": bottom_y, "bbox": (x, y, w, h)
            })

        # BLUE TARGETS
        for contour in contours_blue:
            area = cv2.contourArea(contour)
            if area < 160: continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0: continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.55 or aspect_ratio > 1.9: continue
            if area > 25000: continue
            if x < 5 or x + w > width - 5: continue
            roi = hsv[y:y+h, x:x+w]
            if roi.size == 0: continue
            mean_s = np.mean(roi[:, :, 1])
            mean_v = np.mean(roi[:, :, 2])
            if mean_s < 115 or mean_v < 45: continue

            cx = x + w // 2
            bottom_y = y + h
            targets.append({
                "color": "BLUE", "area": area, "cx": cx, "bottom_y": bottom_y, "bbox": (x, y, w, h)
            })

        for t in targets:
            x, y, w, h = t["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, t["color"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        target_msg = Bool()
        error_msg = Float32()
        area_msg = Float32()
        y_msg = Float32()
        color_msg = String()

        if len(targets) > 0:
            # =========================================================
            # 🌟 V10 神级外挂：台球炸球法 (Billiard Break Logic)
            # =========================================================
            red_targets = [t for t in targets if t["color"] == "RED"]
            blue_targets = [t for t in targets if t["color"] == "BLUE"]
           
            billiard_mode = False
           
            if red_targets and blue_targets:
                best_red = max(red_targets, key=lambda t: t["area"])
                best_blue = max(blue_targets, key=lambda t: t["area"])
               
                # 计算最大红块和最大蓝块的像素距离
                dist = math.hypot(best_red["cx"] - best_blue["cx"], best_red["bottom_y"] - best_blue["bottom_y"])
               
                if dist < 120.0:  # 距离小于 120 判定为死死贴在一起
                    billiard_mode = True
                    # 强行篡改坐标：在图传中心打出警报！
                    cv2.putText(frame, "🚨 BILLIARD MODE ACTIVE 🚨", (int(width/2)-150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
                   
                    if best_blue["cx"] > best_red["cx"]:
                        fake_cx = best_red["cx"] - 240  # 蓝在右，准星往左偏
                    else:
                        fake_cx = best_red["cx"] + 240  # 蓝在左，准星往右偏
                       
                    # 把最大目标的坐标强行换成假坐标 (瞒天过海)
                    target = {
                        "color": "RED",
                        "area": best_red["area"],
                        "cx": fake_cx,
                        "bottom_y": best_red["bottom_y"],
                        "bbox": best_red["bbox"]
                    }
           
            if not billiard_mode:
                # 正常情况：挑全场面积最大的打
                target = max(targets, key=lambda t: t["area"])

            # 计算发给大脑的 error (无论真假，大脑都会执行)
            error = target["cx"] - width / 2

            target_msg.data = True
            error_msg.data = float(error)
            area_msg.data = float(target["area"])
            y_msg.data = float(target["bottom_y"])
            color_msg.data = target["color"]

            self.lost_count = 0

            self.last_target = {
                "found": True,
                "color": target["color"],
                "error": float(error),
                "area": float(target["area"]),
                "y": float(target["bottom_y"])
            }

            # =========================================================
            # 🎨 图传特效绘制 (如果是炸球模式，画个显眼的橙色假准星)
            # =========================================================
            x, y, w, h = target["bbox"]
            aim_cx = target["cx"]
           
            circle_color = (0, 165, 255) if billiard_mode else (0, 255, 255) # 炸球用橙色，正常用黄色
            text_label = "FAKE AIM!" if billiard_mode else "SELECTED"
           
            cv2.circle(frame, (aim_cx, int(target["bottom_y"] - h / 2)), 6, circle_color, -1)
            cv2.putText(frame, text_label, (aim_cx - 40, int(target["bottom_y"] - h / 2) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)

        else:
            self.lost_count += 1

            if self.lost_count <= self.MAX_LOST_FRAMES and self.last_target["found"]:
                target_msg.data = True
                error_msg.data = float(self.last_target["error"])
                area_msg.data = float(self.last_target["area"])
                y_msg.data = float(self.last_target["y"])
                color_msg.data = self.last_target["color"]
            else:
                target_msg.data = False
                error_msg.data = 0.0
                area_msg.data = 0.0
                y_msg.data = 0.0
                color_msg.data = "NONE"
                self.last_target["found"] = False

        # GREEN TAPE DETECTION
        green_msg = Bool()
        green_y_msg = Float32()

        green_detected = False
        green_y = 0.0

        best_green = None
        best_green_area = 0.0

        for c in contours_green:
            area = cv2.contourArea(c)
            if area < 1500: continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 2 * h: continue
            if area > best_green_area:
                best_green_area = area
                best_green = (x, y, w, h)

        if best_green is not None:
            x, y, w, h = best_green
            green_detected = True
            green_y = float(y + h / 2.0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 3)
            cv2.putText(frame, "GREEN LINE", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        green_msg.data = green_detected
        green_y_msg.data = green_y

        # ARUCO DETECTION
        aruco_found_msg = Bool()
        aruco_id_msg = Int32()
        aruco_error_msg = Float32()
        aruco_width_msg = Float32()

        aruco_detected = False
        aruco_id = -1
        aruco_error = 0.0
        aruco_width = 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_new_aruco_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            best_i = 0
            best_width = 0.0

            for i in range(len(ids)):
                pts = corners[i][0]
                width_px = float(np.linalg.norm(pts[0] - pts[1]))
                if width_px > best_width:
                    best_width = width_px
                    best_i = i

            pts = corners[best_i][0]
            aruco_id = int(ids[best_i][0])
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            aruco_error = float(cx - width / 2)
            aruco_width = best_width
            aruco_detected = True

            cv2.aruco.drawDetectedMarkers(frame, [corners[best_i]], ids[best_i:best_i+1])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ARUCO ID:{aruco_id}", (cx - 60, max(30, cy - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        aruco_found_msg.data = aruco_detected
        aruco_id_msg.data = aruco_id
        aruco_error_msg.data = aruco_error
        aruco_width_msg.data = aruco_width

        # PUBLISH EVERYTHING
        self.target_pub.publish(target_msg)
        self.error_pub.publish(error_msg)
        self.area_pub.publish(area_msg)
        self.y_pub.publish(y_msg)
        self.color_pub.publish(color_msg)

        self.green_pub.publish(green_msg)
        self.green_y_pub.publish(green_y_msg)

        self.aruco_pub.publish(aruco_found_msg)
        self.aruco_id_pub.publish(aruco_id_msg)
        self.aruco_error_pub.publish(aruco_error_msg)
        self.aruco_width_pub.publish(aruco_width_msg)

        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        cv2.putText(frame, f"T:{color_msg.data} A:{area_msg.data:.0f} E:{error_msg.data:.0f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"G:{green_detected} Gy:{green_y:.0f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Aruco:{aruco_id} Err:{aruco_error:.0f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        with self.frame_lock:
            self.frame = frame.copy()

        now = time.time()

        if now - self.last_print_time > 0.5:
            self.last_print_time = now
            print("\n===== VISION STATUS =====")
            print(f"Target found : {target_msg.data}")
            print(f"Target color : {color_msg.data}")
            print(f"Target area  : {area_msg.data:.1f}")
            print(f"Target error : {error_msg.data:.1f}")
            print(f"Target y     : {y_msg.data:.1f}")
            print(f"Lost count   : {self.lost_count}")
            print(f"Green line   : {green_detected}, y={green_y:.1f}")
            print(f"ArUco found  : {aruco_detected}, id={aruco_id}, err={aruco_error:.1f}, width={aruco_width:.1f}")
            print("=========================\n")

    def generate_frames(self):
        while True:
            with self.frame_lock:
                frame = None if self.frame is None else self.frame.copy()

            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    jpg = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n\r\n')

            time.sleep(0.03)

    def run_flask(self):
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        self.app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)


def main(args=None):
    rclpy.init(args=args)
    node = ColorTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

