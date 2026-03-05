#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import time
import threading
from flask import Flask, Response, render_template_string
import sys

# ==================== BLUE CUBE DETECTOR CLASS ====================
class BlueCubeDetector:
    def __init__(self):
        # Blue color range (adjust based on your cube and lighting)
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # Cube-specific parameters
        self.min_area = 500
        self.max_area = 30000
        self.solidity_threshold = 0.80
        
        # Performance optimization
        self.resolution = (640, 480)
        
    def detect_cubes(self, frame):
        """
        Main detection method
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
            # Basic area filter
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Analyze candidate
            candidate = self.analyze_candidate(contour)
            
            if candidate:
                candidates.append(candidate)
                
                # Verify if it's a cube
                if self.verify_cube(candidate):
                    verified_cubes.append(candidate)
        
        return verified_cubes, candidates, blue_mask
    
    def analyze_candidate(self, contour):
        """
        Analyze a potential cube candidate
        """
        try:
            # Basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Aspect ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Approximate polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            return {
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'area': area,
                'solidity': solidity,
                'aspect_ratio': aspect_ratio,
                'corner_count': len(approx)
            }
            
        except Exception as e:
            return None
    
    def verify_cube(self, candidate):
        """
        Verify if candidate is a cube
        """
        # Solidity check
        if candidate['solidity'] < self.solidity_threshold:
            return False
        
        # Aspect ratio check
        if candidate['aspect_ratio'] > 2.0:
            return False
        
        # Corner count check
        if candidate['corner_count'] < 3 or candidate['corner_count'] > 10:
            return False
        
        return True
    
    def get_cube_position(self, cube, frame_shape):
        """
        Calculate relative position of cube
        """
        x, y, w, h = cube['bbox']
        frame_center_x = frame_shape[1] // 2
        frame_center_y = frame_shape[0] // 2
        
        cube_center_x = x + w//2
        cube_center_y = y + h//2
        
        # Calculate offset from center (normalized -1 to 1)
        x_offset = (cube_center_x - frame_center_x) / frame_center_x if frame_center_x > 0 else 0
        y_offset = (cube_center_y - frame_center_y) / frame_center_y if frame_center_y > 0 else 0
        
        # Estimate distance based on size
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

# ==================== ROS2 NODE WITH FLASK WEB INTERFACE ====================
class CubeDetectorNode(Node):
    def __init__(self):
        super().__init__('cube_detector_node')
        
        # Initialize bridge and detector
        self.bridge = CvBridge()
        self.detector = BlueCubeDetector()
        
        # Image topic - CHANGE THIS TO MATCH YOUR TURTLEBOT'S CAMERA TOPIC
        self.image_topic = '/camera/image_raw'  # Common topics: /image_raw, /raspicam_node/image
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.listener_callback,
            10)
        
        # Store the latest annotated frame for web streaming
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.detection_stats = {
            'cubes_found': 0,
            'fps': 0,
            'last_detection_time': time.time()
        }
        
        # FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        self.get_logger().info(f'Cube detector node started, subscribed to {self.image_topic}')
        self.get_logger().info('Starting web interface on http://0.0.0.0:5000')
        
        # Start Flask server in a separate thread
        self.flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        self.flask_thread.start()
    
    def listener_callback(self, msg):
        """Process incoming images"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        # Detect cubes
        verified_cubes, candidates, mask = self.detector.detect_cubes(cv_image)
        
        # Create annotated frame
        annotated_frame = self.create_annotated_frame(cv_image, verified_cubes, candidates)
        
        # Update stats
        self.frame_count += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.detection_stats['fps'] = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()
        
        self.detection_stats['cubes_found'] = len(verified_cubes)
        if verified_cubes:
            self.detection_stats['last_detection_time'] = time.time()
        
        # Store frame for web streaming
        with self.frame_lock:
            self.latest_frame = annotated_frame
        
        # Log detections
        if verified_cubes:
            for i, cube in enumerate(verified_cubes):
                pos = self.detector.get_cube_position(cube, cv_image.shape)
                self.get_logger().info(f"Cube {i+1}: X={pos['x_offset']:.2f}, Dist={pos['distance']:.2f}")
    
    def create_annotated_frame(self, frame, verified_cubes, candidates):
        """Create annotated frame with detection visualization"""
        annotated = frame.copy()
        
        # Draw candidates (yellow)
        for candidate in candidates:
            x, y, w, h = candidate['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        # Draw verified cubes (green)
        for cube in verified_cubes:
            x, y, w, h = cube['bbox']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center
            cv2.circle(annotated, cube['center'], 4, (0, 0, 255), -1)
            
            # Add label
            label = f"Cube ({cube['solidity']:.2f})"
            cv2.putText(annotated, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Calculate and show position
            pos = self.detector.get_cube_position(cube, frame.shape)
            info = f"X:{pos['x_offset']:.2f} D:{pos['distance']:.2f}"
            cv2.putText(annotated, info, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add status overlay
        self.add_status_overlay(annotated)
        
        return annotated
    
    def add_status_overlay(self, frame):
        """Add status information to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Add text
        cv2.putText(frame, f"Cubes detected: {self.detection_stats['cubes_found']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {self.detection_stats['fps']}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add web interface URL
        cv2.putText(frame, "Web: http://<turtlebot-ip>:5000", (w - 220, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # ==================== FLASK WEB INTERFACE ====================
    def run_flask(self):
        """Run Flask server in a thread"""
        app = Flask(__name__)
        
        # HTML template for the main page
        HTML_TEMPLATE = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>TurtleBot3 Blue Cube Detection</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f0f0;
                    text-align: center;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                }
                .video-container {
                    margin: 20px 0;
                    background-color: #000;
                    padding: 10px;
                    border-radius: 5px;
                }
                img {
                    max-width: 100%;
                    border: 2px solid #333;
                }
                .stats {
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                }
                .stat-box {
                    text-align: center;
                }
                .stat-label {
                    font-size: 14px;
                    color: #666;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }
                .controls {
                    margin: 20px 0;
                }
                button {
                    padding: 10px 20px;
                    margin: 0 10px;
                    font-size: 16px;
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                button:hover {
                    background-color: #45a049;
                }
                .info {
                    color: #666;
                    font-size: 14px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 TurtleBot3 Blue Cube Detection</h1>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-label">Cubes Detected</div>
                        <div class="stat-value" id="cubeCount">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">FPS</div>
                        <div class="stat-value" id="fps">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Status</div>
                        <div class="stat-value" id="status">Running</div>
                    </div>
                </div>
                
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" width="640" height="480">
                </div>
                
                <div class="controls">
                    <button onclick="location.reload()">Refresh Feed</button>
                </div>
                
                <div class="info">
                    <p>📍 <strong>Detection Info:</strong> Green boxes = verified cubes | Yellow boxes = candidates</p>
                    <p>🔵 Looking for blue cubes (HSV: 100-130)</p>
                    <p>📡 Stream updated in real-time</p>
                </div>
            </div>
            
            <script>
                // Auto-refresh stats every second
                setInterval(function() {
                    fetch('/stats')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('cubeCount').textContent = data.cubes;
                            document.getElementById('fps').textContent = data.fps;
                        });
                }, 1000);
            </script>
        </body>
        </html>
        '''
        
        @app.route('/')
        def index():
            return render_template_string(HTML_TEMPLATE)
        
        @app.route('/video_feed')
        def video_feed():
            def generate():
                while True:
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            # Encode frame as JPEG
                            _, jpeg = cv2.imencode('.jpg', self.latest_frame)
                            frame_bytes = jpeg.tobytes()
                            
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.03)  # Limit to ~30 FPS
            
            return Response(generate(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route('/stats')
        def stats():
            return {
                'cubes': self.detection_stats['cubes_found'],
                'fps': self.detection_stats['fps'],
                'time': time.strftime("%H:%M:%S")
            }
        
        # Run Flask server
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    
    def cleanup(self):
        """Cleanup resources"""
        self.get_logger().info("Cleaning up...")


# ==================== MAIN FUNCTION ====================
def main(args=None):
    rclpy.init(args=args)
    node = CubeDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()