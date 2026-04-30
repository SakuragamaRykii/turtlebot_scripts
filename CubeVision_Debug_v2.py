#!/usr/bin/env python3
import cv2
import numpy as np


def build_red_mask(hsv, bgr):
    lower_red_1 = np.array([0, 110, 55], dtype=np.uint8)
    upper_red_1 = np.array([9, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([172, 110, 55], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red_1, upper_red_1),
        cv2.inRange(hsv, lower_red_2, upper_red_2),
    )
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    rgb_mask = np.zeros_like(hsv_mask)
    rgb_mask[(r >= 90) & (r > g + 36) & (r > b + 52)] = 255
    mask = cv2.bitwise_and(hsv_mask, rgb_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def build_blue_mask(hsv, bgr):
    lower_blue = np.array([107, 115, 40], dtype=np.uint8)
    upper_blue = np.array([126, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    rgb_mask = np.zeros_like(hsv_mask)
    rgb_mask[(b >= 70) & (b > g + 20) & (b > r + 45)] = 255
    mask = cv2.bitwise_and(hsv_mask, rgb_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def largest_box(mask, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 260:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if best is None or area > best[0]:
            best = (area, x, y, w, h, color_name)
    return best


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    print('q = quit')
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = build_red_mask(hsv, frame)
        blue_mask = build_blue_mask(hsv, frame)

        best_red = largest_box(red_mask, 'red')
        best_blue = largest_box(blue_mask, 'blue')
        best = best_blue if best_blue is not None else best_red

        view = frame.copy()
        if best is not None:
            _, x, y, w, h, color_name = best
            color = (255, 0, 0) if color_name == 'blue' else (0, 0, 255)
            cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
            cv2.putText(view, color_name, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('view', view)
        cv2.imshow('red_mask', red_mask)
        cv2.imshow('blue_mask', blue_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
