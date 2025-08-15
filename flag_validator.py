import cv2
import numpy as np
from PIL import Image
import io
import json
from color_analysis import ColorAnalyzer
from chakra_detector import ChakraDetector
from image_processor import ImageProcessor

class FlagValidator:
    """Championship-grade Indian Flag validator with dual-algorithm validation"""
    
    def __init__(self, debug_mode=False, competition_mode=True):
        self.debug_mode = debug_mode
        self.competition_mode = competition_mode
        self.debug_image = None
        
        # Initialize components
        self.color_analyzer = ColorAnalyzer(competition_mode)
        self.chakra_detector = ChakraDetector(debug_mode)
        self.image_processor = ImageProcessor(debug_mode)
        
        # Competition mode uses tighter tolerances
        self.aspect_ratio_tolerance = 0.001 if competition_mode else 0.01  # 0.1% vs 1%
        
    def validate_flag(self, image_bytes):
        """Main validation function with progressive checking"""
        
        # Load and preprocess image
        image = self.image_processor.load_image(image_bytes)
        processed_image = self.image_processor.preprocess_image(image)
        
        result = {}
        
        # 1. Aspect Ratio Check (fail-fast)
        result['aspect_ratio'] = self._check_aspect_ratio(processed_image)
        
        # 2. Color Analysis
        result['colors'] = self.color_analyzer.analyze_colors(processed_image)
        
        # 3. Stripe Proportion Analysis
        result['stripe_proportion'] = self._check_stripe_proportions(processed_image)
        
        # 4. Chakra Analysis
        chakra_results = self.chakra_detector.detect_chakra(processed_image)
        result['chakra_position'] = chakra_results['position']
        result['chakra_spokes'] = chakra_results['spokes']
        
        # Generate debug visualization if enabled
        if self.debug_mode:
            self.debug_image = self._create_debug_visualization(processed_image, result)
        
        return result
    
    def _check_aspect_ratio(self, image):
        """Check aspect ratio with sub-pixel precision"""
        height, width = image.shape[:2]
        actual_ratio = width / height
        expected_ratio = 1.5  # 3:2
        
        deviation = abs(actual_ratio - expected_ratio) / expected_ratio
        status = 'pass' if deviation <= self.aspect_ratio_tolerance else 'fail'
        
        # Calculate confidence based on how close to ideal ratio
        confidence = max(0, 100 - (deviation * 10000))  # Scale for display
        
        return {
            'status': status,
            'actual': f"{actual_ratio:.4f}",
            'expected': f"{expected_ratio:.4f}",
            'deviation': f"{deviation*100:.2f}%",
            'confidence': f"{confidence:.1f}%"
        }
    
    def _check_stripe_proportions(self, image):
        """Check if each stripe occupies exactly 1/3 of the height"""
        height, width = image.shape[:2]
        
        # Divide image into three horizontal bands
        stripe_height = height // 3
        
        # Calculate actual proportions
        top_prop = stripe_height / height
        middle_prop = stripe_height / height  
        bottom_prop = (height - 2 * stripe_height) / height
        
        # Check if proportions are close to 1/3 each
        tolerance = 0.02 if self.competition_mode else 0.05  # 2% vs 5%
        expected = 1/3
        
        top_ok = abs(top_prop - expected) <= tolerance
        middle_ok = abs(middle_prop - expected) <= tolerance  
        bottom_ok = abs(bottom_prop - expected) <= tolerance
        
        status = 'pass' if (top_ok and middle_ok and bottom_ok) else 'fail'
        
        return {
            'status': status,
            'top': f"{top_prop:.3f}",
            'middle': f"{middle_prop:.3f}",
            'bottom': f"{bottom_prop:.3f}",
            'expected': f"{expected:.3f}"
        }
    
    def _create_debug_visualization(self, image, results):
        """Create debug visualization showing detection results"""
        debug_img = image.copy()
        height, width = debug_img.shape[:2]
        
        # Draw stripe boundaries
        stripe_height = height // 3
        cv2.line(debug_img, (0, stripe_height), (width, stripe_height), (255, 255, 255), 2)
        cv2.line(debug_img, (0, 2 * stripe_height), (width, 2 * stripe_height), (255, 255, 255), 2)
        
        # Draw center lines
        cv2.line(debug_img, (width//2, 0), (width//2, height), (255, 255, 255), 1)
        cv2.line(debug_img, (0, height//2), (width, height//2), (255, 255, 255), 1)
        
        # Add text annotations
        cv2.putText(debug_img, "SAFFRON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, "WHITE", (10, stripe_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(debug_img, "GREEN", (10, 2 * stripe_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_img
