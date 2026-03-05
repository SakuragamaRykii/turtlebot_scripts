#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
# ... (rest of your BlueCubeDetector class remains the same) ...

class BlueCubeDetector:
    def __init__(self):
        # Blue color range (adjust as needed)
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # Cube-specific parameters
        self.min_area = 1000
        self.max_area = 50000  # Maximum expected cube size
        self.solidity_threshold = 0.82  # Cubes are very solid
        self.aspect_ratio_range = (0.7, 1.3)  # Square-like shape
        
        # 3D cue parameters
        self.min_face_count = 2  # Minimum visible faces for cube
        self.edge_ratio_range = (0.5, 2.0)  # Range for perspective distortion
        
    def detect_cubes(self, frame):
        """
        Main detection method with multiple verification steps
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create blue mask
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        verified_cubes = []
        candidates = []
        
        for contour in contours:
            # Step 1: Basic area filter
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Step 2: Create analysis structure
            cube_candidate = self.analyze_candidate(contour, frame)
            
            if cube_candidate:
                candidates.append(cube_candidate)
                
                # Step 3: Apply cube-specific verification
                if self.verify_cube(cube_candidate, frame):
                    verified_cubes.append(cube_candidate)
        
        return verified_cubes, candidates, blue_mask
    
    def analyze_candidate(self, contour, frame):
        """
        Comprehensive analysis of a candidate region
        """
        # Basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Fit ellipse (for shape analysis)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
        else:
            ellipse = None
            axis_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            center = (x + w//2, y + h//2)
        
        # Approximate polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Corner detection within the region
        corners = self.detect_corners(frame, (x, y, w, h))
        
        # Edge detection and analysis
        edges = self.detect_edges(frame, (x, y, w, h))
        
        # Texture analysis
        texture_score = self.analyze_texture(frame, (x, y, w, h))
        
        return {
            'contour': contour,
            'approx': approx,
            'bbox': (x, y, w, h),
            'center': center,
            'area': area,
            'perimeter': perimeter,
            'solidity': solidity,
            'extent': extent,
            'axis_ratio': axis_ratio,
            'ellipse': ellipse,
            'corner_count': len(corners),
            'edge_count': len(edges),
            'texture_score': texture_score,
            'corners': corners,
            'edges': edges
        }
    
    def verify_cube(self, candidate, frame):
        """
        Multi-factor cube verification
        """
        x, y, w, h = candidate['bbox']
        
        # Factor 1: Solidity (cubes are solid, not complex shapes)
        if candidate['solidity'] < self.solidity_threshold:
            return False
        
        # Factor 2: Aspect ratio (should be roughly square)
        aspect = w / h if w > h else h / w
        if aspect > 1.3:  # Too elongated
            return False
        
        # Factor 3: Corner analysis
        # Real cubes have distinct corners, monitors have smooth edges
        if candidate['corner_count'] < 4 or candidate['corner_count'] > 8:
            # Too few or too many corners suggests non-cube object
            pass  # Don't fail immediately, use as soft indicator
        
        # Factor 4: Edge analysis
        # Cubes typically have strong edges in multiple orientations
        edge_strength = self.analyze_edge_orientation(candidate['edges'])
        if edge_strength < 0.3:  # Weak edges suggest non-cube
            return False
        
        # Factor 5: Texture analysis
        # Monitors have smooth, uniform texture; cubes have more varied texture
        if candidate['texture_score'] < 0.1:  # Too smooth
            # Check if it might be a monitor
            if self.is_likely_monitor(candidate, frame):
                return False
        
        # Factor 6: Color uniformity
        color_variance = self.analyze_color_uniformity(frame, candidate['bbox'])
        if color_variance < 5:  # Too uniform (like a monitor screen)
            # Could be a solid color screen
            if self.is_likely_monitor(candidate, frame):
                return False
        
        # Factor 7: 3D cues (look for multiple faces)
        if w > 30 and h > 30:  # Only if region is large enough
            if self.has_multiple_faces(candidate, frame):
                return True
        
        # If we've passed most checks, it's likely a cube
        return True
    
    def detect_corners(self, frame, bbox):
        """
        Detect corners within the bounding box using Shi-Tomasi
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return []
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, 
                                         qualityLevel=0.01, 
                                         minDistance=10)
        
        if corners is not None:
            # Convert to global coordinates
            corners = corners.reshape(-1, 2)
            corners[:, 0] += x
            corners[:, 1] += y
            return corners
        return []
    
    def detect_edges(self, frame, bbox):
        """
        Detect edges within the bounding box
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return []
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find edge contours
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert to global coordinates
        valid_edges = []
        for edge in edge_contours:
            if cv2.contourArea(edge) > 50:
                edge[:, :, 0] += x
                edge[:, :, 1] += y
                valid_edges.append(edge)
        
        return valid_edges
    
    def analyze_edge_orientation(self, edges):
        """
        Analyze the distribution of edge orientations
        """
        if len(edges) < 3:
            return 0
        
        orientations = []
        for edge in edges:
            if len(edge) >= 2:
                # Simplified orientation analysis
                pts = edge.reshape(-1, 2)
                if len(pts) >= 2:
                    vec = pts[-1] - pts[0]
                    angle = math.atan2(vec[1], vec[0])
                    orientations.append(angle)
        
        if len(orientations) < 2:
            return 0
        
        # Check if we have edges in multiple orientations
        orientations = np.array(orientations)
        unique_orientations = len(np.unique(np.round(orientations, 1)))
        
        return min(unique_orientations / 3, 1.0)  # Normalize
    
    def analyze_texture(self, frame, bbox):
        """
        Analyze texture using variance and edge density
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture metrics
        variance = np.var(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        # Combine metrics
        texture_score = (variance / 10000) * 0.5 + edge_density * 0.5
        
        return min(texture_score, 1.0)
    
    def analyze_color_uniformity(self, frame, bbox):
        """
        Analyze color variance within the region
        """
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 100  # High variance as fallback
        
        # Calculate standard deviation of each channel
        stds = [np.std(roi[:, :, i]) for i in range(3)]
        mean_std = np.mean(stds)
        
        return mean_std
    
    def is_likely_monitor(self, candidate, frame):
        """
        Check if the detected object is likely a monitor
        """
        x, y, w, h = candidate['bbox']
        
        # Monitors typically have:
        # 1. Very straight edges
        # 2. Uniform texture
        # 3. Specific aspect ratios (16:9, 4:3, etc.)
        
        # Check aspect ratio
        aspect = w / h
        monitor_aspects = [1.33, 1.78, 1.6, 1.25]  # Common monitor ratios
        aspect_match = any(abs(aspect - ma) < 0.2 for ma in monitor_aspects)
        
        # Check texture uniformity
        if candidate['texture_score'] < 0.2 and aspect_match:
            return True
        
        # Check for straight edges
        if len(candidate['edges']) >= 2:
            # Monitor typically has 4 straight edges
            straight_edges = 0
            for edge in candidate['edges'][:4]:  # Check first few edges
                if len(edge) >= 10:
                    # Check if edge is relatively straight
                    pts = edge.reshape(-1, 2)
                    if len(pts) >= 2:
                        vec = pts[-1] - pts[0]
                        edge_len = np.linalg.norm(vec)
                        if edge_len > 20:  # Long enough edge
                            straight_edges += 1
            
            if straight_edges >= 3 and aspect_match:
                return True
        
        return False
    
    def has_multiple_faces(self, candidate, frame):
        """
        Detect if multiple faces of a cube are visible
        """
        x, y, w, h = candidate['bbox']
        
        # Look for internal edges that suggest 3D structure
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Use Hough lines to find potential edges between faces
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                threshold=50, 
                                minLineLength=20, 
                                maxLineGap=10)
        
        if lines is None:
            return False
        
        # Analyze line orientations
        orientations = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1)
            orientations.append(abs(angle))
        
        # Check if we have lines in at least 2 distinct orientations
        if len(orientations) > 2:
            unique_orientations = len(np.unique(np.round(orientations, 1)))
            return unique_orientations >= 2
        
        return False

class CubeDetectorNode(Node):
    def __init__(self):
        super().__init__('cube_detector_node')
        
        # ... (existing setup code) ...
        
        # Add debug control variables
        self.show_candidates = False
        self.frame_count = 0
        self.last_print_time = time.time()
        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.fps = 0
        
        # Initialize detector
        self.detector = BlueCubeDetector()
        
        # Add keyboard handler thread
        import threading
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        
        # Print instructions
        self.print_instructions()
        
    def print_instructions(self):
        """Print control instructions"""
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info("TurtleBot 3 Blue Cube Detection")
        self.get_logger().info("="*50)
        self.get_logger().info("Controls:")
        self.get_logger().info("  'q' - Quit")
        self.get_logger().info("  'd' - Toggle candidate display")
        self.get_logger().info("  'p' - Print cube positions")
        self.get_logger().info("  's' - Save current frame")
        self.get_logger().info("  'h' - Show help")
        self.get_logger().info("="*50)
    
    def keyboard_listener(self):
        """Listen for keyboard input in a separate thread"""
        import sys
        import select
        import tty
        import termios
        
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        try:
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self.handle_keypress(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def handle_keypress(self, key):
        """Handle keyboard commands"""
        if key == 'q':
            self.get_logger().info("Shutting down...")
            rclpy.shutdown()
        elif key == 'd':
            self.show_candidates = not self.show_candidates
            self.get_logger().info(f"Show candidates: {self.show_candidates}")
        elif key == 'p' and hasattr(self, 'last_cube_positions'):
            self.get_logger().info("\n[CUBE POSITIONS]")
            for i, pos in enumerate(self.last_cube_positions):
                self.get_logger().info(f"  Cube {i+1}:")
                self.get_logger().info(f"    X Offset: {pos['x_offset']:.3f}")
                self.get_logger().info(f"    Y Offset: {pos['y_offset']:.3f}")
                self.get_logger().info(f"    Distance: {pos['distance']:.2f}")
                self.get_logger().info(f"    Center: {pos['center']}")
                self.get_logger().info(f"    Size: {pos['size']}")
        elif key == 's' and hasattr(self, 'last_display_frame'):
            filename = f"cube_detection_{self.frame_count}.jpg"
            cv2.imwrite(filename, self.last_display_frame)
            self.get_logger().info(f"Saved {filename}")
            self.frame_count += 1
        elif key == 'h':
            self.print_instructions()
    
    def add_debug_info(self, frame, verified_cubes, candidates, fps):
        """Add debug information to the display frame"""
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add cube counts
        cv2.putText(frame, f"Cubes: {len(verified_cubes)}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Candidates: {len(candidates)}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add mode indicator
        mode_text = "Mode: CANDIDATES ON" if self.show_candidates else "Mode: VERIFIED ONLY"
        cv2.putText(frame, mode_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def listener_callback(self, msg):
        """Modified callback with full debug info"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        # Calculate FPS
        self.frame_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_counter / (current_time - self.last_fps_time)
            self.frame_counter = 0
            self.last_fps_time = current_time
        
        # Detect cubes
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        verified_cubes, candidates, mask = self.detector.detect_cubes(frame_rgb)
        
        # Store for keyboard commands
        self.last_cube_positions = []
        
        # Create debug visualization
        display_frame = cv_image.copy()  # Start with BGR image
        
        # Draw candidates if enabled
        if self.show_candidates:
            for candidate in candidates:
                x, y, w, h = candidate['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                # Add candidate info
                cv2.putText(display_frame, f"C:{candidate['corner_count']}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Draw verified cubes
        for cube in verified_cubes:
            x, y, w, h = cube['bbox']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(display_frame, cube['center'], 4, (0, 0, 255), -1)
            
            # Calculate position
            pos = self.get_cube_position(cube, display_frame.shape)
            self.last_cube_positions.append(pos)
            
            # Add detailed info
            info1 = f"S:{cube['solidity']:.2f} C:{cube['corner_count']}"
            info2 = f"X:{pos['x_offset']:.2f} D:{pos['distance']:.2f}"
            
            cv2.putText(display_frame, info1, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            cv2.putText(display_frame, info2, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add debug overlay
        display_frame = self.add_debug_info(display_frame, verified_cubes, candidates, self.fps)
        
        # Store for save command
        self.last_display_frame = display_frame
        
        # Show frame
        cv2.imshow('TurtleBot3 Cube Detection', display_frame)
        cv2.waitKey(1)
        
        # Periodic logging
        if self.last_cube_positions and time.time() - self.last_print_time > 2.0:
            for i, pos in enumerate(self.last_cube_positions):
                self.get_logger().info(f"Cube {i+1}: X={pos['x_offset']:.3f}, Dist={pos['distance']:.2f}")
            self.last_print_time = time.time()
    
    def get_cube_position(self, cube, frame_shape):
        """Calculate relative position (from your original script)"""
        x, y, w, h = cube['bbox']
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        cube_center_x = x + w//2
        cube_center_y = y + h//2
        
        x_offset = (cube_center_x - frame_center_x) / frame_center_x
        y_offset = (cube_center_y - frame_center_y) / frame_center_y
        
        max_expected_area = 20000
        distance_estimate = 1.0 - min(cube['area'] / max_expected_area, 1.0)
        
        return {
            'x_offset': x_offset,
            'y_offset': y_offset,
            'distance': distance_estimate,
            'center': (cube_center_x, cube_center_y),
            'size': (w, h),
            'area': cube['area']
        }

def main(args=None):
    rclpy.init(args=args)
    node = CubeDetectorNode()
    
    # Keep the node alive and processing callbacks
    rclpy.spin(node)
    
    # Clean shutdown
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()