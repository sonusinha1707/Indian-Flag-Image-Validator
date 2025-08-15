import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys

class ColorAnalyzer:
    """Advanced color analysis with multiple color spaces"""
    
    def __init__(self, competition_mode=True):
        self.competition_mode = competition_mode
        
        # BIS specified colors (RGB)
        self.target_colors = {
            'saffron': (255, 153, 51),   # #FF9933
            'white': (255, 255, 255),    # #FFFFFF
            'green': (19, 136, 8),       # #138808
            'chakra_blue': (0, 0, 128)   # #000080
        }
        
        # Convert to other color spaces for analysis
        self._prepare_color_spaces()
    
    def _prepare_color_spaces(self):
        """Convert target colors to HSV and LAB color spaces"""
        self.target_hsv = {}
        self.target_lab = {}
        
        for name, rgb in self.target_colors.items():
            # Convert to HSV
            rgb_norm = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)  # BGR for OpenCV
            hsv = cv2.cvtColor(rgb_norm, cv2.COLOR_BGR2HSV)
            self.target_hsv[name] = hsv[0, 0]
            
            # Convert to LAB
            lab = cv2.cvtColor(rgb_norm, cv2.COLOR_BGR2LAB)
            self.target_lab[name] = lab[0, 0]
    
    def analyze_colors(self, image):
        """Comprehensive color analysis using multiple algorithms"""
        height, width = image.shape[:2]
        
        # Define regions for each color
        stripe_height = height // 3
        
        regions = {
            'saffron': image[0:stripe_height, :],
            'white': image[stripe_height:2*stripe_height, :],
            'green': image[2*stripe_height:, :],
        }
        
        results = {}
        
        # Analyze each stripe color
        for color_name, region in regions.items():
            results[color_name] = self._analyze_region_color(region, color_name)
        
        # Analyze chakra color (from center of white region)
        white_region = regions['white']
        chakra_region = self._extract_chakra_region(white_region)
        results['chakra_blue'] = self._analyze_chakra_color(chakra_region)
        
        return results
    
    def _extract_chakra_region(self, white_region):
        """Extract the chakra region from white stripe"""
        h, w = white_region.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Extract circular region around center
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        chakra_region = white_region.copy()
        chakra_region[~mask] = [255, 255, 255]  # Mask out non-chakra areas
        
        return chakra_region
    
    def _analyze_region_color(self, region, color_name):
        """Analyze color of a specific region using dual algorithms"""
        
        # Algorithm 1: Dominant color using K-means
        dominant_color = self._get_dominant_color_kmeans(region)
        
        # Algorithm 2: Average color with outlier removal
        avg_color = self._get_average_color_robust(region)
        
        # Cross-verify results
        target_rgb = self.target_colors[color_name]
        
        # Calculate deviations for both methods
        deviation_dominant = self._calculate_color_deviation(dominant_color, target_rgb)
        deviation_avg = self._calculate_color_deviation(avg_color, target_rgb)
        
        # Use the better result (lower deviation)
        if deviation_dominant < deviation_avg:
            final_color = dominant_color
            final_deviation = deviation_dominant
            method = "dominant"
        else:
            final_color = avg_color
            final_deviation = deviation_avg
            method = "average"
        
        # Determine pass/fail
        tolerance = 0.05  # 5% tolerance as per specification
        status = 'pass' if final_deviation <= tolerance else 'fail'
        
        # Calculate confidence
        confidence = max(0, 100 - (final_deviation * 2000))  # Scale for display
        
        return {
            'status': status,
            'deviation': f"{final_deviation*100:.1f}%",
            'detected_rgb': final_color,
            'expected_rgb': target_rgb,
            'method': method,
            'confidence': f"{confidence:.1f}%"
        }
    
    def _analyze_chakra_color(self, chakra_region):
        """Special analysis for chakra color (navy blue)"""
        
        # Convert to HSV to better detect blue regions
        hsv = cv2.cvtColor(chakra_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue regions (navy blue range in HSV)
        lower_blue = np.array([100, 50, 20])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        if np.sum(blue_mask) == 0:
            # No blue detected
            return {
                'status': 'fail',
                'deviation': '100.0%',
                'detected_rgb': [0, 0, 0],
                'expected_rgb': self.target_colors['chakra_blue'],
                'confidence': '0.0%'
            }
        
        # Extract blue pixels
        blue_pixels = chakra_region[blue_mask > 0]
        avg_blue = np.mean(blue_pixels, axis=0).astype(int)
        
        # Calculate deviation
        target_rgb = self.target_colors['chakra_blue']
        deviation = self._calculate_color_deviation(avg_blue, target_rgb)
        
        tolerance = 0.05
        status = 'pass' if deviation <= tolerance else 'fail'
        confidence = max(0, 100 - (deviation * 2000))
        
        return {
            'status': status,
            'deviation': f"{deviation*100:.1f}%",
            'detected_rgb': avg_blue.tolist(),
            'expected_rgb': target_rgb,
            'confidence': f"{confidence:.1f}%"
        }
    
    def _get_dominant_color_kmeans(self, region):
        """Get dominant color using K-means clustering"""
        # Reshape image to pixel array
        pixels = region.reshape(-1, 3)
        
        # Remove very dark and very bright pixels (likely noise/edges)
        mask = np.all((pixels > 10) & (pixels < 245), axis=1)
        clean_pixels = pixels[mask]
        
        if len(clean_pixels) == 0:
            return np.mean(pixels, axis=0).astype(int)
        
        # Apply K-means with k=3 to find dominant colors
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        kmeans.fit(clean_pixels)
        
        # Get the cluster with most points
        labels = kmeans.labels_
        if labels is not None:
            counts = np.bincount(labels)
        else:
            counts = np.array([1, 0, 0])  # Fallback
        dominant_cluster = np.argmax(counts)
        
        return kmeans.cluster_centers_[dominant_cluster].astype(int)
    
    def _get_average_color_robust(self, region):
        """Get average color with outlier removal"""
        pixels = region.reshape(-1, 3)
        
        # Remove outliers using IQR method
        for channel in range(3):
            q25 = np.percentile(pixels[:, channel], 25)
            q75 = np.percentile(pixels[:, channel], 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Keep pixels within bounds
            mask = (pixels[:, channel] >= lower_bound) & (pixels[:, channel] <= upper_bound)
            pixels = pixels[mask]
        
        return np.mean(pixels, axis=0).astype(int)
    
    def _calculate_color_deviation(self, color1, color2):
        """Calculate color deviation using Delta-E in LAB space"""
        
        # Convert colors to LAB space for perceptual accuracy
        color1_bgr = np.array([[[color1[2], color1[1], color1[0]]]], dtype=np.uint8)
        color2_bgr = np.array([[[color2[2], color2[1], color2[0]]]], dtype=np.uint8)
        
        lab1 = cv2.cvtColor(color1_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        lab2 = cv2.cvtColor(color2_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        
        # Calculate Delta-E (CIE76)
        delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2))
        
        # Normalize to 0-1 range (100 is max Delta-E for practical purposes)
        normalized_deviation = min(delta_e / 100.0, 1.0)
        
        return normalized_deviation
