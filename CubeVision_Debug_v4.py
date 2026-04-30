#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass

import cv2
import numpy as np


MIN_CONTOUR_AREA = 260.0
MIN_BBOX_W = 15
MIN_BBOX_H = 15
MIN_ASPECT = 0.52
MAX_ASPECT = 1.90
MIN_FILL_RATIO = 0.25
MIN_EXTENT = 0.21
MIN_SOLIDITY = 0.68
MIN_CENTER_Y_RATIO = 0.22
BOTTOM_FLOOR_BONUS_RATIO = 0.45


@dataclass
class CubeObservation:
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


class CubeDetector:
    def __init__(self):
        self.image_width = 0
        self.image_height = 0

    def build_red_mask(self, hsv, bgr):
        lower_red_1 = np.array([0, 95, 45], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([168, 95, 45], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_mask = np.zeros_like(hsv_mask)
        red_dom = (r >= 70) & (r > g + 28) & (r > b + 28)
        rgb_mask[red_dom] = 255
        return cv2.bitwise_and(hsv_mask, rgb_mask)

    def build_blue_mask(self, hsv, bgr):
        lower_blue_core = np.array([100, 90, 35], dtype=np.uint8)
        upper_blue_core = np.array([130, 255, 255], dtype=np.uint8)
        lower_blue_wide = np.array([95, 75, 30], dtype=np.uint8)
        upper_blue_wide = np.array([138, 255, 255], dtype=np.uint8)
        hsv_core = cv2.inRange(hsv, lower_blue_core, upper_blue_core)
        hsv_wide = cv2.inRange(hsv, lower_blue_wide, upper_blue_wide)

        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        rgb_core = np.zeros_like(hsv_core)
        rgb_wide = np.zeros_like(hsv_core)
        rgb_core[(b >= 60) & (b > r + 22) & (b > g + 12)] = 255
        rgb_wide[(b >= 55) & (b > r + 18) & (b > g - 8)] = 255
        return cv2.bitwise_or(cv2.bitwise_and(hsv_core, rgb_core), cv2.bitwise_and(hsv_wide, rgb_wide))

    def preprocess_mask(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def color_signature(self, color, roi_bgr, roi_hsv, roi_mask):
        pixels = roi_bgr[roi_mask > 0]
        hsv_pixels = roi_hsv[roi_mask > 0]
        if pixels.size == 0 or hsv_pixels.size == 0:
            return False, 0.0

        mean_b, mean_g, mean_r = np.mean(pixels, axis=0)
        _, mean_s, mean_v = np.mean(hsv_pixels, axis=0)
        hue = hsv_pixels[:, 0]
        sat = hsv_pixels[:, 1]

        if color == 'blue':
            hue_ratio = float(np.mean((hue >= 104) & (hue <= 126) & (sat >= 105)))
            dom_br = float(mean_b - mean_r)
            dom_bg = float(mean_b - mean_g)
            conf = 0.30 * float(mean_s) + 0.30 * dom_br + 0.20 * dom_bg + 70.0 * hue_ratio
            ok = hue_ratio >= 0.48 and mean_s >= 120.0 and mean_v >= 45.0 and mean_b >= 65.0 and dom_br >= 34.0 and dom_bg >= 8.0
            return ok, float(conf)

        hue_ratio = float(np.mean(((hue <= 13) | (hue >= 168)) & (sat >= 105)))
        dom_rb = float(mean_r - mean_b)
        dom_rg = float(mean_r - mean_g)
        conf = 0.30 * float(mean_s) + 0.30 * dom_rb + 0.20 * dom_rg + 70.0 * hue_ratio
        ok = hue_ratio >= 0.50 and mean_s >= 125.0 and mean_v >= 55.0 and mean_r >= 82.0 and dom_rb >= 38.0 and dom_rg >= 24.0
        return ok, float(conf)

    def detect_holes(self, gray_roi, shape_mask_roi):
        if gray_roi.size == 0 or shape_mask_roi.size == 0:
            return 0, 0.0

        inner_mask = cv2.erode(shape_mask_roi, np.ones((5, 5), np.uint8), iterations=1)
        valid_pixels = gray_roi[inner_mask > 0]
        if valid_pixels.size < 30:
            return 0, 0.0

        dark_threshold = int(np.clip(np.percentile(valid_pixels, 22), 18, 95))
        dark_mask = cv2.inRange(gray_roi, 0, dark_threshold)
        dark_mask = cv2.bitwise_and(dark_mask, inner_mask)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_area = max(float(np.count_nonzero(inner_mask)), 1.0)
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(7.0, 0.00025 * mask_area) or area > 0.026 * mask_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < 0.22:
                continue
            m = cv2.moments(cnt)
            if m['m00'] == 0:
                continue
            points.append((float(m['m10'] / m['m00']), float(m['m01'] / m['m00'])))

        if len(points) < 2:
            return len(points), 0.0
        arr = np.array(points, dtype=np.float32)
        nearest = []
        for i in range(len(arr)):
            d = np.linalg.norm(arr - arr[i], axis=1)
            d = d[d > 1.0]
            if d.size > 0:
                nearest.append(float(np.min(d)))
        return len(points), float(np.median(nearest)) if nearest else 0.0

    def detect_best_for_color(self, color, color_mask, frame, hsv, gray):
        mask = self.preprocess_mask(color_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_BBOX_W or h < MIN_BBOX_H:
                continue

            cy = y + h / 2.0
            if cy < self.image_height * MIN_CENTER_Y_RATIO:
                continue

            aspect = w / float(h)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue

            bbox_area = float(w * h)
            hull = cv2.convexHull(contour)
            solidity = float(area / max(cv2.contourArea(hull), 1.0))
            if solidity < MIN_SOLIDITY:
                continue

            shifted = contour - np.array([[x, y]])
            shape_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(shape_mask, [shifted], -1, 255, thickness=-1)
            fill_ratio = float(np.count_nonzero(shape_mask) / max(bbox_area, 1.0))
            extent = float(area / max(bbox_area, 1.0))
            if fill_ratio < MIN_FILL_RATIO or extent < MIN_EXTENT:
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
            center_error = abs((x + w / 2.0) - self.image_width / 2.0)
            bottom = y + h
            floor_bonus = 180.0 if bottom >= self.image_height * BOTTOM_FLOOR_BONUS_RATIO else 0.0
            high_penalty = 260.0 if cy < self.image_height * 0.34 and h < self.image_height * 0.20 else 0.0
            score = (
                1.85 * area
                + 430.0 * fill_ratio
                + 330.0 * extent
                + 260.0 * solidity
                + 2.0 * color_conf
                + 12.0 * min(holes, 10)
                + 2.0 * hole_pitch
                + floor_bonus
                - 0.18 * center_error
                - high_penalty
            )

            obs = CubeObservation(color, x + w / 2.0, cy, area, x, y, w, h, holes, hole_pitch, fill_ratio, extent, solidity, color_conf, score)
            if best is None or obs.score > best.score:
                best = obs
        return best

    def detect(self, frame):
        self.image_height, self.image_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        red_raw = self.build_red_mask(hsv, frame)
        blue_raw = self.build_blue_mask(hsv, frame)
        red_mask = self.preprocess_mask(red_raw)
        blue_mask = self.preprocess_mask(blue_raw)

        blue_best = self.detect_best_for_color('blue', blue_raw, frame, hsv, gray)
        red_best = self.detect_best_for_color('red', red_raw, frame, hsv, gray)
        chosen = blue_best if blue_best is not None else red_best
        return chosen, red_mask, blue_mask


def draw_target(frame, obs):
    if obs is None:
        cv2.putText(frame, 'target: none', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        return
    color = (255, 80, 20) if obs.color == 'blue' else (30, 30, 255)
    x, y, w, h = obs.bbox_x, obs.bbox_y, obs.bbox_w, obs.bbox_h
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.circle(frame, (int(obs.cx), int(obs.cy)), 4, color, -1)
    label = f'{obs.color} area={obs.area:.0f} h={obs.bbox_h} holes={obs.holes} pitch={obs.hole_pitch:.1f}'
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    detector = CubeDetector()

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        obs, red_mask, blue_mask = detector.detect(frame)
        display = frame.copy()
        draw_target(display, obs)

        cv2.imshow('original + chosen target', display)
        cv2.imshow('red mask', red_mask)
        cv2.imshow('blue mask', blue_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q'), ord('h'), ord('H')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
