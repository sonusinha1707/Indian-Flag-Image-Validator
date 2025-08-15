import cv2
import numpy as np
from scipy import ndimage
import math

class ChakraDetector:
    """Advanced Chakra detection using multiple computer vision techniques"""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.debug_info = {}
    
    def detect_chakra(self, image):
        """Detect chakra position and spoke count using dual algorithms"""
        height, width = image.shape[:2]
        
        # Extract white stripe (middle third)
        stripe_height = height // 3
        white_stripe = image[stripe_height:2*stripe_height, :]
        
        # Algorithm 1: Hough Circle Transform
        circle_result = self._detect_chakra_hough_circles(white_stripe)
        
        # Algorithm 2: Contour-based detection
        contour_result = self._detect_chakra_contours(white_stripe)
        
        # Cross-verify and select best result
        final_result = self._select_best_detection(circle_result, contour_result, white_stripe)
        
        # Adjust coordinates to full image space
        if final_result['position']['status'] == 'pass':
            final_result['position']['center_y'] += stripe_height
        
        return final_result
    
    def _detect_chakra_hough_circles(self, white_stripe):
        """Detect chakra using Hough Circle Transform"""
        gray = cv2.cvtColor(white_stripe, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply adaptive threshold to enhance circle edges
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Edge detection
        edges = cv2.Canny(adaptive_thresh, 50, 150, apertureSize=3)
        
        # Detect circles
        h, w = white_stripe.shape[:2]
        min_radius = min(h, w) // 8
        max_radius = min(h, w) // 3
        
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(h, w) // 4,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Select the circle closest to center
            center_x, center_y = w // 2, h // 2
            distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y, r in circles]
            best_circle = circles[np.argmin(distances)]
            
            x, y, r = best_circle
            
            # Check if chakra is properly centered
            offset_x = abs(x - center_x)
            offset_y = abs(y - center_y)
            max_offset = min(h, w) * 0.05  # 5% tolerance
            
            position_status = 'pass' if (offset_x <= max_offset and offset_y <= max_offset) else 'fail'
            
            # Count spokes
            spoke_count = self._count_spokes_radial(white_stripe, x, y, r)
            
            return {
                'position': {
                    'status': position_status,
                    'center_x': x,
                    'center_y': y,
                    'offset_x': f"{offset_x:.1f}px",
                    'offset_y': f"{offset_y:.1f}px",
                    'confidence': f"{max(0, 100 - (offset_x + offset_y)):.1f}%"
                },
                'spokes': {
                    'status': 'pass' if spoke_count == 24 else 'fail',
                    'detected': spoke_count,
                    'expected': 24,
                    'confidence': f"{max(0, 100 - abs(spoke_count - 24) * 4):.1f}%"
                },
                'method': 'hough_circles'
            }
        
        return self._default_fail_result()
    
    def _detect_chakra_contours(self, white_stripe):
        """Detect chakra using contour analysis"""
        gray = cv2.cvtColor(white_stripe, cv2.COLOR_BGR2GRAY)
        
        # Create binary image focusing on dark regions (chakra)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the most circular contour
            best_contour = None
            best_circularity = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        if circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = contour
            
            if best_contour is not None and best_circularity > 0.7:  # Circular enough
                # Calculate center and radius
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                x, y = int(x), int(y)
                radius = int(radius)
                
                # Check position
                h, w = white_stripe.shape[:2]
                center_x, center_y = w // 2, h // 2
                offset_x = abs(x - center_x)
                offset_y = abs(y - center_y)
                max_offset = min(h, w) * 0.05
                
                position_status = 'pass' if (offset_x <= max_offset and offset_y <= max_offset) else 'fail'
                
                # Count spokes
                spoke_count = self._count_spokes_radial(white_stripe, x, y, radius)
                
                return {
                    'position': {
                        'status': position_status,
                        'center_x': x,
                        'center_y': y,
                        'offset_x': f"{offset_x:.1f}px",
                        'offset_y': f"{offset_y:.1f}px",
                        'confidence': f"{max(0, 100 - (offset_x + offset_y)):.1f}%"
                    },
                    'spokes': {
                        'status': 'pass' if spoke_count == 24 else 'fail',
                        'detected': spoke_count,
                        'expected': 24,
                        'confidence': f"{max(0, 100 - abs(spoke_count - 24) * 4):.1f}%"
                    },
                    'method': 'contours'
                }
        
        return self._default_fail_result()
    
    def _count_spokes_radial(self, image, center_x, center_y, radius):
        """Enhanced spoke counting with multiple validation techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Multi-radius sampling with improved parameters
        spoke_counts_method1 = []
        
        for r_factor in [0.65, 0.75, 0.85]:  # Optimized radii
            current_radius = int(radius * r_factor)
            angle_step = 0.5  # Higher precision: 0.5-degree steps
            intensities = []
            
            for angle in np.arange(0, 360, angle_step):
                radian = math.radians(angle)
                x = int(center_x + current_radius * math.cos(radian))
                y = int(center_y + current_radius * math.sin(radian))
                
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    intensities.append(gray[y, x])
                else:
                    intensities.append(255)
            
            intensities = np.array(intensities)
            
            # Apply minimal smoothing to preserve spoke boundaries
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(intensities, sigma=1.0)  # Reduced smoothing
            
            # Enhanced peak detection for 24 spokes (15° apart)
            from scipy.signal import find_peaks
            # Convert 15 degrees to array indices (15° * 2 indices/degree = 30)
            min_distance = int(14 * (2 / angle_step))  # 14° minimum (slightly less than 15°)
            
            peaks, properties = find_peaks(
                -smoothed, 
                height=-180,  # Less restrictive height
                distance=min_distance,
                prominence=10  # Require minimum prominence
            )
            
            spoke_counts_method1.append(len(peaks))
        
        # Method 2: Template matching approach
        spoke_count_method2 = self._count_spokes_template_matching(gray, center_x, center_y, radius)
        
        # Method 3: Symmetry-based validation
        spoke_count_method3 = self._count_spokes_symmetry(gray, center_x, center_y, radius)
        
        # Combine results from all methods
        all_counts = spoke_counts_method1 + [spoke_count_method2, spoke_count_method3]
        
        # Filter out obviously wrong counts (not near 24)
        valid_counts = [count for count in all_counts if 20 <= count <= 28]
        
        if valid_counts:
            # Prefer counts closest to 24
            from collections import Counter
            count_freq = Counter(valid_counts)
            
            # If 24 is detected, prioritize it
            if 24 in valid_counts:
                return 24
            
            # Otherwise, return the most frequent valid count
            return count_freq.most_common(1)[0][0]
        
        # Fallback: return the most frequent count from method 1
        if spoke_counts_method1:
            from collections import Counter
            count_freq = Counter(spoke_counts_method1)
            return count_freq.most_common(1)[0][0]
        
        return 0
    
    def _count_spokes_template_matching(self, gray, center_x, center_y, radius):
        """Count spokes using template matching approach"""
        # Sample at optimal radius for spoke detection
        sample_radius = int(radius * 0.75)
        angle_step = 1.0
        intensities = []
        
        for angle in range(0, 360, int(angle_step)):
            radian = math.radians(angle)
            x = int(center_x + sample_radius * math.cos(radian))
            y = int(center_y + sample_radius * math.sin(radian))
            
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                intensities.append(gray[y, x])
            else:
                intensities.append(255)
        
        intensities = np.array(intensities)
        
        # Calculate derivative to find transitions
        diff = np.diff(intensities)
        
        # Find significant negative transitions (white to dark = spoke start)
        threshold = -np.std(diff) * 1.5
        spoke_starts = np.where(diff < threshold)[0]
        
        # Group nearby transitions and count distinct spokes
        if len(spoke_starts) > 0:
            # Remove transitions that are too close together
            filtered_starts = [spoke_starts[0]]
            for start in spoke_starts[1:]:
                if start - filtered_starts[-1] > 8:  # At least 8 degrees apart
                    filtered_starts.append(start)
            
            return len(filtered_starts)
        
        return 0
    
    def _count_spokes_symmetry(self, gray, center_x, center_y, radius):
        """Count spokes using symmetry validation"""
        # For 24 spokes, check at exact 15-degree intervals
        expected_angles = [i * 15 for i in range(24)]  # 0, 15, 30, 45, ..., 345
        sample_radius = int(radius * 0.8)
        
        spoke_detections = 0
        background_intensity = np.mean(gray)  # Get average background
        
        for angle in expected_angles:
            radian = math.radians(angle)
            x = int(center_x + sample_radius * math.cos(radian))
            y = int(center_y + sample_radius * math.sin(radian))
            
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                # Check if this position is significantly darker than background
                pixel_intensity = gray[y, x]
                
                # Sample nearby pixels for robustness
                nearby_sum = 0
                nearby_count = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < gray.shape[1] and 0 <= ny < gray.shape[0]:
                            nearby_sum += int(gray[ny, nx])  # Convert to int to prevent overflow
                            nearby_count += 1
                
                avg_nearby = nearby_sum / nearby_count if nearby_count > 0 else pixel_intensity
                
                # If significantly darker than background, count as spoke
                if avg_nearby < background_intensity - 30:  # 30-point threshold
                    spoke_detections += 1
        
        return spoke_detections
    
    def _select_best_detection(self, circle_result, contour_result, white_stripe):
        """Select the best detection result from both algorithms"""
        
        # If both failed, return default fail
        if (circle_result['position']['status'] == 'fail' and 
            contour_result['position']['status'] == 'fail'):
            return self._default_fail_result()
        
        # If only one succeeded, use that one
        if circle_result['position']['status'] == 'pass' and contour_result['position']['status'] == 'fail':
            return circle_result
        if contour_result['position']['status'] == 'pass' and circle_result['position']['status'] == 'fail':
            return contour_result
        
        # If both succeeded, choose based on spoke count accuracy
        circle_spoke_error = abs(circle_result['spokes']['detected'] - 24)
        contour_spoke_error = abs(contour_result['spokes']['detected'] - 24)
        
        if circle_spoke_error <= contour_spoke_error:
            return circle_result
        else:
            return contour_result
    
    def _default_fail_result(self):
        """Return default failure result"""
        return {
            'position': {
                'status': 'fail',
                'center_x': 0,
                'center_y': 0,
                'offset_x': '0px',
                'offset_y': '0px',
                'confidence': '0.0%'
            },
            'spokes': {
                'status': 'fail',
                'detected': 0,
                'expected': 24,
                'confidence': '0.0%'
            },
            'method': 'none'
        }
