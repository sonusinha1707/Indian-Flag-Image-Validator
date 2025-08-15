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
        """Count spokes using radial sampling technique"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sample at multiple radii for robustness
        spoke_counts = []
        
        for r_factor in [0.6, 0.7, 0.8, 0.9]:  # Different radii
            current_radius = int(radius * r_factor)
            angle_step = 1  # 1-degree steps for high precision
            intensities = []
            
            for angle in range(0, 360, angle_step):
                radian = math.radians(angle)
                x = int(center_x + current_radius * math.cos(radian))
                y = int(center_y + current_radius * math.sin(radian))
                
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    intensities.append(gray[y, x])
                else:
                    intensities.append(255)  # White if outside image
            
            # Find peaks (dark regions - spokes)
            intensities = np.array(intensities)
            
            # Smooth the signal
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(intensities, sigma=2)
            
            # Find local minima (dark spokes)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(-smoothed, height=-200, distance=8)  # At least 8 degrees apart
            
            spoke_counts.append(len(peaks))
        
        # Use the most consistent count
        if spoke_counts:
            # Return the mode or most frequent count
            from collections import Counter
            count_freq = Counter(spoke_counts)
            return count_freq.most_common(1)[0][0]
        
        return 0
    
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
