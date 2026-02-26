import cv2
import numpy as np
from scipy import spatial
import math

class BlueCubeDetector:
    def __init__(self):
        # Blue color range (adjust as needed)
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # Cube-specific parameters
        self.min_area = 1000
        self.max_area = 50000  # Maximum expected cube size
        self.solidity_threshold = 0.85  # Cubes are very solid
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

def main():
    # Initialize detector
    detector = BlueCubeDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Advanced Blue Cube Detection")
    print("============================")
    print("Controls:")
    print("  'q' - Quit")
    print("  'd' - Show all candidates (including non-cubes)")
    print("  'c' - Show only verified cubes")
    print("  'h' - Show help")
    
    show_candidates = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Detect cubes
        verified_cubes, candidates, mask = detector.detect_cubes(frame)
        
        # Visualize results
        if show_candidates:
            # Show all candidates in yellow
            for candidate in candidates:
                x, y, w, h = candidate['bbox']
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display_frame, f"Solidity: {candidate['solidity']:.2f}", 
                           (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Show verified cubes in green
        for cube in verified_cubes:
            x, y, w, h = cube['bbox']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw contour
            cv2.drawContours(display_frame, [cube['contour']], -1, (255, 0, 0), 2)
            
            # Draw corners if available
            if len(cube['corners']) > 0:
                for corner in cube['corners']:
                    cv2.circle(display_frame, tuple(corner.astype(int)), 4, (0, 0, 255), -1)
            
            # Add label with metrics
            label = f"Cube (S:{cube['solidity']:.2f} C:{cube['corner_count']})"
            cv2.putText(display_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add info text
        cv2.putText(display_frame, f"Cubes: {len(verified_cubes)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Candidates: {len(candidates)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show mask in corner
        mask_small = cv2.resize(mask, (160, 120))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        display_frame[10:130, display_frame.shape[1]-170:display_frame.shape[1]-10] = mask_colored
        
        cv2.imshow('Advanced Cube Detection', display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_candidates = not show_candidates
            print(f"Show candidates: {show_candidates}")
        elif key == ord('h'):
            print("\nControls:")
            print("  'q' - Quit")
            print("  'd' - Toggle candidate display")
            print("  'c' - Show only cubes")
            print("  'h' - Show help\n")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()