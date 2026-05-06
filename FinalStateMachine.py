#!/usr/bin/env python3
"""
Cube Sorting Robot with Reference-Based Color Detection
Uses proven HSV+RGB dual masking from working reference code
"""

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
import json
import os

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None

# ============================================================================
# Reference-Based Color Detection Parameters (from working code)
# ============================================================================

class ColorDetectionConfig:
    """Exact parameters from the working reference code"""
    
    # Red HSV ranges
    RED_HSV_LOWER1 = np.array([0, 75, 40], dtype=np.uint8)
    RED_HSV_UPPER1 = np.array([16, 255, 255], dtype=np.uint8)
    RED_HSV_LOWER2 = np.array([164, 75, 40], dtype=np.uint8)
    RED_HSV_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)
    
    # Red RGB constraints (BGR order)
    RED_R_MIN = 70
    RED_R_OVER_G = 22
    RED_R_OVER_B = 26
    
    # Blue HSV ranges
    BLUE_HSV_LOWER = np.array([80, 25, 20], dtype=np.uint8)
    BLUE_HSV_UPPER = np.array([150, 255, 255], dtype=np.uint8)
    
    # Blue RGB constraints (BGR order)
    BLUE_B_MIN = 40
    BLUE_B_OVER_R = 6
    BLUE_B_OVER_G = -22  # Note: negative means B can be slightly less than G
    
    # Morphological processing
    MEDIAN_BLUR_SIZE = 5
    MORPH_OPEN_KERNEL = (3, 3)
    MORPH_CLOSE_KERNEL = (7, 7)
    DILATE_KERNEL = (3, 3)
    DILATE_ITERATIONS = 1
    
    # Shape filtering
    MIN_CONTOUR_AREA = 120.0
    MIN_BBOX_W = 10
    MIN_BBOX_H = 10
    MIN_ASPECT = 0.55
    MAX_ASPECT = 1.75
    MIN_FILL_RATIO = 0.15
    MIN_EXTENT = 0.13
    MIN_SOLIDITY = 0.40
    MIN_CENTER_Y_RATIO = 0.12
    
    # Hole detection (for cube verification)
    RED_MIN_HOLES = 1
    BLUE_MIN_HOLES = 0
    HOLE_ERODE_KERNEL = (5, 5)
    HOLE_MIN_AREA_RATIO = 0.00015
    HOLE_MAX_AREA_RATIO = 0.06
    HOLE_MIN_CIRCULARITY = 0.14
    HOLE_DARK_PERCENTILE = 30
    HOLE_MIN_PIXELS_FOR_STATS = 20
    
    # Color signature verification
    BLUE_HUE_RANGE = (80, 150)
    BLUE_HUE_MIN_RATIO = 0.18
    BLUE_MIN_SAT = 25
    BLUE_MIN_V = 30
    BLUE_MIN_B = 40
    BLUE_MIN_B_OVER_R = 5
    
    RED_HUE_MIN = 0
    RED_HUE_MAX = 16
    RED_HUE_MIN2 = 164
    RED_HUE_MAX2 = 180
    RED_HUE_MIN_RATIO = 0.25
    RED_MIN_SAT = 65
    RED_MIN_V = 40
    RED_MIN_R = 65
    RED_MIN_R_OVER_B = 20
    RED_MIN_R_OVER_G = 16
    
    # Distance estimation
    REAL_CUBE_SIZE_CM = 5.0
    FOCAL_LENGTH_PX = 520.0


@dataclass
class CubeObservation:
    """Data class for cube detection results"""
    color: str
    cx: float
    cy: float
    area: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    holes: int
    hole_pitch: float
    fill_ratio: float
    extent: float
    solidity: float
    color_conf: float
    score: float
    distance_cm: float = 0.0


class ReferenceCubeDetector:
    """
    Cube detector using proven HSV+RGB dual masking from reference code.
    This should match the color detection performance of the working robot.
    """
    
    def __init__(self):
        self.image_width = 0
        self.image_height = 0
        self.focal_length_px = ColorDetectionConfig.FOCAL_LENGTH_PX
        self.cfg = ColorDetectionConfig()
    
    def build_red_mask(self, hsv, bgr):
        """Build red mask using HSV+RGBT dual approach (from reference)"""
        # HSV masking (two ranges for red)
        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.cfg.RED_HSV_LOWER1, self.cfg.RED_HSV_UPPER1),
            cv2.inRange(hsv, self.cfg.RED_HSV_LOWER2, self.cfg.RED_HSV_UPPER2),
        )
        
        # RGB dominance check
        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        rgb_mask = np.zeros_like(hsv_mask)
        rgb_mask[(r >= self.cfg.RED_R_MIN) & 
                 (r > g + self.cfg.RED_R_OVER_G) & 
                 (r > b + self.cfg.RED_R_OVER_B)] = 255
        
        # Combine both masks
        return cv2.bitwise_and(hsv_mask, rgb_mask)
    
    def build_blue_mask(self, hsv, bgr):
        """Build blue mask using HSV+RGB dual approach (from reference)"""
        # HSV masking
        hsv_mask = cv2.inRange(hsv, self.cfg.BLUE_HSV_LOWER, self.cfg.BLUE_HSV_UPPER)
        
        # RGB dominance check
        b = bgr[:, :, 0].astype(np.int16)
        g = bgr[:, :, 1].astype(np.int16)
        r = bgr[:, :, 2].astype(np.int16)
        rgb_mask = np.zeros_like(hsv_mask)
        rgb_mask[(b >= self.cfg.BLUE_B_MIN) & 
                 (b > r + self.cfg.BLUE_B_OVER_R) & 
                 (b > g + self.cfg.BLUE_B_OVER_G)] = 255
        
        # OR combination for blue (more permissive)
        return cv2.bitwise_or(hsv_mask, rgb_mask)
    
    def preprocess_mask(self, mask):
        """Apply morphological operations to clean up mask (from reference)"""
        mask = cv2.medianBlur(mask, self.cfg.MEDIAN_BLUR_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                                np.ones(self.cfg.MORPH_OPEN_KERNEL, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                                np.ones(self.cfg.MORPH_CLOSE_KERNEL, np.uint8))
        mask = cv2.dilate(mask, 
                         np.ones(self.cfg.DILATE_KERNEL, np.uint8), 
                         iterations=self.cfg.DILATE_ITERATIONS)
        return mask
    
    def detect_holes(self, gray_roi, shape_mask_roi):
        """Detect holes (indentations) characteristic of cubes (from reference)"""
        if gray_roi.size == 0 or shape_mask_roi.size == 0:
            return 0, 0.0
        
        # Create inner mask to avoid edge artifacts
        inner_mask = cv2.erode(shape_mask_roi, 
                               np.ones(self.cfg.HOLE_ERODE_KERNEL, np.uint8), 
                               iterations=1)
        valid_pixels = gray_roi[inner_mask > 0]
        
        if valid_pixels.size < self.cfg.HOLE_MIN_PIXELS_FOR_STATS:
            return 0, 0.0
        
        # Adaptive dark threshold
        dark_threshold = int(np.clip(
            np.percentile(valid_pixels, self.cfg.HOLE_DARK_PERCENTILE), 18, 130
        ))
        
        # Find dark regions
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, 
                                     np.ones((3, 3), np.uint8))
        
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        mask_area = max(float(np.count_nonzero(inner_mask)), 1.0)
        points = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by size relative to mask
            if area < max(4.0, self.cfg.HOLE_MIN_AREA_RATIO * mask_area):
                continue
            if area > self.cfg.HOLE_MAX_AREA_RATIO * mask_area:
                continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue
            
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < self.cfg.HOLE_MIN_CIRCULARITY:
                continue
            
            moments = cv2.moments(cnt)
            if moments["m00"] == 0:
                continue
            
            points.append((
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"])
            ))
        
        # Calculate hole pitch (distance between holes)
        if len(points) < 2:
            return len(points), 0.0
        
        arr = np.array(points, dtype=np.float32)
        nearest = []
        for i in range(len(arr)):
            d = np.linalg.norm(arr - arr[i], axis=1)
            d = d[d > 1.0]  # Avoid zero distance to self
            if d.size > 0:
                nearest.append(float(np.min(d)))
        
        return len(points), float(np.median(nearest)) if nearest else 0.0
    
    def color_signature(self, color, roi_bgr, roi_hsv, roi_mask):
        """Verify color consistency in detected region (from reference)"""
        pixels = roi_bgr[roi_mask > 0]
        hsv_pixels = roi_hsv[roi_mask > 0]
        
        if pixels.size == 0 or hsv_pixels.size == 0:
            return False, 0.0
        
        mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
        _, mean_s, mean_v = np.mean(hsv_pixels, axis=0)
        hue = hsv_pixels[:, 0]
        sat = hsv_pixels[:, 1]
        
        if color == "blue":
            hue_ratio = float(np.mean(
                (hue >= self.cfg.BLUE_HUE_RANGE[0]) & 
                (hue <= self.cfg.BLUE_HUE_RANGE[1]) & 
                (sat >= self.cfg.BLUE_MIN_SAT)
            ))
            dom_br = float(mean_b - mean_r)
            dom_bg = float(mean_b - mean_g)
            
            conf = 0.25 * mean_s + 0.35 * dom_br + 0.15 * dom_bg + 90.0 * hue_ratio
            ok = (hue_ratio >= self.cfg.BLUE_HUE_MIN_RATIO and 
                  mean_v >= self.cfg.BLUE_MIN_V and 
                  mean_b >= self.cfg.BLUE_MIN_B and 
                  dom_br >= self.cfg.BLUE_MIN_B_OVER_R)
            
            return ok, float(conf)
        
        else:  # red
            hue_ratio = float(np.mean(
                ((hue <= self.cfg.RED_HUE_MAX) | (hue >= self.cfg.RED_HUE_MIN2)) & 
                (sat >= self.cfg.RED_MIN_SAT)
            ))
            dom_rb = float(mean_r - mean_b)
            dom_rg = float(mean_r - mean_g)
            
            conf = 0.25 * mean_s + 0.35 * dom_rb + 0.25 * dom_rg + 90.0 * hue_ratio
            ok = (hue_ratio >= self.cfg.RED_HUE_MIN_RATIO and 
                  mean_s >= self.cfg.RED_MIN_SAT and 
                  mean_v >= self.cfg.RED_MIN_V and 
                  mean_r >= self.cfg.RED_MIN_R and 
                  dom_rb >= self.cfg.RED_MIN_R_OVER_B and 
                  dom_rg >= self.cfg.RED_MIN_R_OVER_G)
            
            return ok, float(conf)
    
    def estimate_distance_cm(self, obs):
        """Estimate distance to cube based on apparent size"""
        apparent_size_px = (obs.bbox_w + obs.bbox_h) / 2.0
        if apparent_size_px <= 0:
            return 0.0
        return float(self.cfg.REAL_CUBE_SIZE_CM * self.focal_length_px / apparent_size_px)
    
    def detect_best_for_color(self, color, color_mask, frame, hsv, gray):
        """Find the best cube candidate for a given color (from reference)"""
        mask = self.preprocess_mask(color_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.cfg.MIN_CONTOUR_AREA:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.cfg.MIN_BBOX_W or h < self.cfg.MIN_BBOX_H:
                continue
            if w > self.image_width * 0.45 or h > self.image_height * 0.60:
                continue
            
            cy = y + h / 2.0
            if cy < self.image_height * self.cfg.MIN_CENTER_Y_RATIO:
                continue
            
            aspect = w / float(h)
            if aspect < self.cfg.MIN_ASPECT or aspect > self.cfg.MAX_ASPECT:
                continue
            
            bbox_area = float(w * h)
            hull = cv2.convexHull(contour)
            solidity = float(area / max(cv2.contourArea(hull), 1.0))
            if solidity < self.cfg.MIN_SOLIDITY:
                continue
            
            shifted = contour - np.array([[x, y]])
            shape_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(shape_mask, [shifted], -1, 255, thickness=-1)
            fill_ratio = float(np.count_nonzero(shape_mask) / max(bbox_area, 1.0))
            extent = float(area / max(bbox_area, 1.0))
            
            if fill_ratio < self.cfg.MIN_FILL_RATIO or extent < self.cfg.MIN_EXTENT:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
            if len(approx) < 4 or len(approx) > 12:
                continue
            
            roi_bgr = frame[y:y + h, x:x + w]
            roi_hsv = hsv[y:y + h, x:x + w]
            roi_gray = gray[y:y + h, x:x + w]
            color_ok, color_conf = self.color_signature(color, roi_bgr, roi_hsv, shape_mask)
            if not color_ok:
                continue
            
            holes, hole_pitch = self.detect_holes(roi_gray, shape_mask)
            min_holes = self.cfg.RED_MIN_HOLES if color == "red" else self.cfg.BLUE_MIN_HOLES
            if holes < min_holes:
                continue
            
            center_error = abs((x + w / 2.0) - self.image_width / 2.0)
            # Scoring function from reference
            score = (
                1.7 * area +
                420.0 * fill_ratio +
                320.0 * extent +
                240.0 * solidity +
                2.2 * color_conf +
                80.0 * min(holes, 12) +
                3.0 * hole_pitch -
                0.16 * center_error
            )
            
            obs = CubeObservation(
                color=color,
                cx=x + w / 2.0,
                cy=cy,
                area=area,
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
                holes=holes,
                hole_pitch=hole_pitch,
                fill_ratio=fill_ratio,
                extent=extent,
                solidity=solidity,
                color_conf=color_conf,
                score=score,
            )
            obs.distance_cm = self.estimate_distance_cm(obs)
            
            if best is None or obs.score > best.score:
                best = obs
        
        return best
    
    def detect(self, frame):
        """Main detection method - returns best cube (from reference)"""
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect both colors
        red_best = self.detect_best_for_color(
            "red", self.build_red_mask(hsv, frame), frame, hsv, gray
        ) if TARGET_COLOR in ("any", "red") else None
        
        blue_best = self.detect_best_for_color(
            "blue", self.build_blue_mask(hsv, frame), frame, hsv, gray
        ) if TARGET_COLOR in ("any", "blue") else None
        
        # Return based on target preference
        if TARGET_COLOR == "red":
            return red_best, red_best, blue_best
        if TARGET_COLOR == "blue":
            return blue_best, red_best, blue_best
        
        # For "any" target, prefer based on PREFER_BLUE setting
        if red_best is not None and blue_best is not None:
            chosen = blue_best if PREFER_BLUE and blue_best.score >= red_best.score else red_best
        else:
            chosen = blue_best if blue_best is not None else red_best
        
        return chosen, red_best, blue_best
    
    def draw_debug(self, frame, chosen, red_obs, blue_obs):
        """Draw detection visualization for debugging"""
        debug_frame = frame.copy()
        
        # Draw red detection
        if red_obs is not None:
            x, y, w, h = red_obs.bbox_x, red_obs.bbox_y, red_obs.bbox_w, red_obs.bbox_h
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(debug_frame, f"RED s={red_obs.score:.0f} h={red_obs.holes}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw blue detection
        if blue_obs is not None:
            x, y, w, h = blue_obs.bbox_x, blue_obs.bbox_y, blue_obs.bbox_w, blue_obs.bbox_h
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_frame, f"BLUE s={blue_obs.score:.0f} h={blue_obs.holes}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Highlight chosen cube
        if chosen is not None:
            x, y, w, h = chosen.bbox_x, chosen.bbox_y, chosen.bbox_w, chosen.bbox_h
            cv2.rectangle(debug_frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 3)
            cv2.putText(debug_frame, f"TARGET: {chosen.color}", 
                       (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return debug_frame


# ============================================================================
# Quick Test Script
# ============================================================================

def test_cube_detection():
    """Standalone test to verify color detection works"""
    print("Starting Cube Detection Test...")
    print("Using reference-based HSV+RGB dual masking")
    print("Press 'q' to quit, 's' to save test image\n")
    
    detector = ReferenceCubeDetector()
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try other indices
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        print("Trying to load test image...")
        # Try to load a test image
        test_img = cv2.imread("test_cubes.jpg")
        if test_img is None:
            print("No test image found. Please place cubes in view and run again.")
            return
        
        # Process single image
        chosen, red_obs, blue_obs = detector.detect(test_img)
        debug = detector.draw_debug(test_img, chosen, red_obs, blue_obs)
        
        if chosen:
            print(f"Detected: {chosen.color} cube")
            print(f"  Score: {chosen.score:.1f}")
            print(f"  Holes: {chosen.holes}")
            print(f"  Distance: {chosen.distance_cm:.1f} cm")
            print(f"  BBox: {chosen.bbox_w}x{chosen.bbox_h}")
        else:
            print("No cube detected in test image")
        
        cv2.imshow("Detection Test", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    
    print("Camera opened successfully!")
    frame_count = 0
    detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Run detection
        chosen, red_obs, blue_obs = detector.detect(frame)
        
        if chosen is not None:
            detections += 1
        
        # Draw results
        debug = detector.draw_debug(frame, chosen, red_obs, blue_obs)
        
        # Add stats
        cv2.putText(debug, f"Frames: {frame_count} | Detections: {detections}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if chosen:
            cv2.putText(debug, f"Target: {chosen.color} | Dist: {chosen.distance_cm:.1f}cm", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Reference-Based Cube Detection", debug)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"detection_test_{frame_count}.jpg", frame)
            print(f"Saved frame {frame_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    if frame_count > 0:
        print(f"\nDetection Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with detection: {detections}")
        print(f"  Detection rate: {100*detections/frame_count:.1f}%")


# Global settings matching reference
TARGET_COLOR = "any"
PREFER_BLUE = True

if __name__ == "__main__":
    print("=" * 60)
    print("Cube Detection Test - Using Reference Color Parameters")
    print("=" * 60)
    print(f"TARGET_COLOR: {TARGET_COLOR}")
    print(f"PREFER_BLUE: {PREFER_BLUE}")
    print()
    print("This uses the EXACT HSV+RGB masking from your working code:")
    print("  RED: HSV([0,75,40]-[16,255,255] OR [164,75,40]-[180,255,255])")
    print("       AND RGB(r>=70, r>g+22, r>b+26)")
    print("  BLUE: HSV([80,25,20]-[150,255,255]) OR RGB(b>=40, b>r+6, b>g-22)")
    print("=" * 60)
    print()
    
    test_cube_detection()