#!/usr/bin/env python3

import cv2
import numpy as np
import math
import time
import os
import sys
from picamera2 import Picamera2
import signal

class BlueCubeDetector:
    def __init__(self):
        # Blue color range (adjust based on your cube and lighting)
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # Cube-specific parameters
        self.min_area = 500  # Reduced for PiCamera (lower resolution)
        self.max_area = 30000  # Adjusted for typical PiCamera resolution
        self.solidity_threshold = 0.80  # Slightly lowered for PiCamera quality
        
        # 3D cue parameters
        self.min_face_count = 2
        self.edge_ratio_range = (0.5, 2.0)
        
        # Performance optimization for Pi
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0
        self.resolution = (640, 480)  # Good balance for Pi
        
        # Debug and logging
        self.debug_mode = False
        self.last_detection_time = time.time()
        
    def setup_camera(self):
        """
        Initialize the PiCamera with optimal settings for detection
        """
        try:
            picam2 = Picamera2()
            
            # Configure camera for video capture
            config = picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={
                    "FrameDurationLimits": (33333, 33333),  # ~30fps
                    "AwbEnable": True,  # Auto white balance
                    "AeEnable": True,   # Auto exposure
                }
            )
            
            picam2.configure(config)
            picam2.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            print(f"[INFO] Camera initialized at {self.resolution}")
            return picam2
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera: {e}")
            return None
    
    def preprocess_frame(self, frame):
        """
        Optimize frame for detection on Pi
        """
        # Resize if needed (already set in config)
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Convert to HSV (required for color detection)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        
        return hsv
    
    def detect_cubes(self, frame):
        """
        Main detection method optimized for Pi
        """
        # Preprocess frame
        hsv = self.preprocess_frame(frame)
        
        # Create blue mask
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        # Use smaller kernel for Pi (less processing)
        kernel = np.ones((3,3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (optimized for Pi)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        verified_cubes = []
        candidates = []
        
        for contour in contours:
            # Basic area filter
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Simplified analysis for speed
            cube_candidate = self.analyze_candidate_fast(contour, frame)
            
            if cube_candidate:
                candidates.append(cube_candidate)
                
                # Quick verification
                if self.verify_cube_fast(cube_candidate):
                    verified_cubes.append(cube_candidate)
        
        return verified_cubes, candidates, blue_mask
    
    def analyze_candidate_fast(self, contour, frame):
        """
        Faster analysis for Pi
        """
        try:
            # Basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convex hull (simplified)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Aspect ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Approximate polygon (simplified)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Quick corner estimate
            corner_count = len(approx)
            
            return {
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'corner_count': corner_count
            }
            
        except Exception as e:
            return None
    
    def verify_cube_fast(self, candidate):
        """
        Fast verification for Pi
        """
        # Solidity check (primary indicator)
        if candidate['solidity'] < self.solidity_threshold:
            return False
        
        # Aspect ratio check
        if candidate['aspect_ratio'] > 2.0:  # Too elongated
            return False
        
        # Corner count check (cubes typically have 4-8 corners)
        if candidate['corner_count'] < 3 or candidate['corner_count'] > 10:
            return False
        
        return True
    
    def get_cube_position(self, cube, frame_shape):
        """
        Calculate relative position of cube for robot navigation
        """
        x, y, w, h = cube['bbox']
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        cube_center_x = x + w//2
        cube_center_y = y + h//2
        
        # Calculate offset from center (normalized -1 to 1)
        x_offset = (cube_center_x - frame_center_x) / frame_center_x
        y_offset = (cube_center_y - frame_center_y) / frame_center_y
        
        # Estimate distance based on size (rough estimate)
        # Larger area = closer
        max_expected_area = 20000  # Adjust based on your cube and camera
        distance_estimate = 1.0 - min(cube['area'] / max_expected_area, 1.0)
        
        return {
            'x_offset': x_offset,
            'y_offset': y_offset,
            'distance': distance_estimate,
            'center': (cube_center_x, cube_center_y),
            'size': (w, h),
            'area': cube['area']
        }
    
    def draw_detections(self, frame, verified_cubes, candidates, show_candidates):
        """
        Draw detections on frame (optimized)
        """
        display = frame.copy()
        
        # Convert RGB to BGR for OpenCV display
        if len(display.shape) == 3 and display.shape[2] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        
        # Show candidates if enabled
        if show_candidates:
            for candidate in candidates:
                x, y, w, h = candidate['bbox']
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        # Show verified cubes
        cube_positions = []
        for cube in verified_cubes:
            x, y, w, h = cube['bbox']
            
            # Draw bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(display, cube['center'], 4, (0, 0, 255), -1)
            
            # Add label
            label = f"Cube ({cube['solidity']:.2f})"
            cv2.putText(display, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Calculate position for robot
            pos = self.get_cube_position(cube, frame.shape)
            cube_positions.append(pos)
            
            # Draw position info
            info = f"X:{pos['x_offset']:.2f} D:{pos['distance']:.2f}"
            cv2.putText(display, info, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return display, cube_positions

def signal_handler(sig, frame):
    """
    Handle Ctrl+C gracefully
    """
    print("\n[INFO] Shutting down...")
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize detector
    detector = BlueCubeDetector()
    
    # Setup camera
    print("[INFO] Initializing PiCamera...")
    picam2 = detector.setup_camera()
    
    if picam2 is None:
        print("[ERROR] Failed to initialize camera")
        return
    
    print("\n" + "="*50)
    print("TurtleBot 3 Blue Cube Detection")
    print("="*50)
    print("Controls:")
    print("  'q' - Quit")
    print("  'd' - Toggle candidate display")
    print("  'p' - Print cube positions")
    print("  's' - Save current frame")
    print("  'h' - Show help")
    print("="*50)
    
    show_candidates = False
    frame_count = 0
    last_print_time = time.time()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            if frame is None:
                print("[ERROR] Failed to capture frame")
                continue
            
            # Detect cubes
            verified_cubes, candidates, mask = detector.detect_cubes(frame)
            
            # Draw detections
            display_frame, cube_positions = detector.draw_detections(
                frame, verified_cubes, candidates, show_candidates
            )
            
            # Add status info
            fps = detector.frame_count / (time.time() - detector.last_detection_time + 0.001)
            cv2.putText(display_frame, f"Cubes: {len(verified_cubes)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('TurtleBot3 Cube Detection', display_frame)
            
            # Print cube positions periodically
            if cube_positions and time.time() - last_print_time > 1.0:
                for i, pos in enumerate(cube_positions):
                    print(f"[INFO] Cube {i+1}: Offset X={pos['x_offset']:.3f}, "
                          f"Distance={pos['distance']:.2f}")
                last_print_time = time.time()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_candidates = not show_candidates
                print(f"[INFO] Show candidates: {show_candidates}")
            elif key == ord('p') and cube_positions:
                for i, pos in enumerate(cube_positions):
                    print(f"\n[CUBE {i+1}] Position Data:")
                    print(f"  X Offset: {pos['x_offset']:.3f}")
                    print(f"  Y Offset: {pos['y_offset']:.3f}")
                    print(f"  Distance: {pos['distance']:.2f}")
                    print(f"  Center: {pos['center']}")
                    print(f"  Size: {pos['size']}")
            elif key == ord('s'):
                filename = f"cube_detection_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[INFO] Saved {filename}")
                frame_count += 1
            elif key == ord('h'):
                print("\nControls:")
                print("  'q' - Quit")
                print("  'd' - Toggle candidate display")
                print("  'p' - Print cube positions")
                print("  's' - Save current frame")
                print("  'h' - Show this help\n")
            
            detector.frame_count += 1
            
    except Exception as e:
        print(f"[ERROR] {e}")
    
    finally:
        # Cleanup
        print("[INFO] Cleaning up...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()