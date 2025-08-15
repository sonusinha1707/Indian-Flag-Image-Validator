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
        """Enhanced color analysis with multiple validation techniques"""
        
        # Algorithm 1: Multi-sample K-means clustering
        dominant_color = self._get_dominant_color_enhanced_kmeans(region)
        
        # Algorithm 2: Weighted average with intelligent outlier removal
        avg_color = self._get_weighted_average_color(region)
        
        # Algorithm 3: Mode-based color detection
        mode_color = self._get_mode_color(region)
        
        # Algorithm 4: Histogram peak analysis
        hist_color = self._get_histogram_peak_color(region)
        
        # Cross-verify results with target color
        target_rgb = self.target_colors[color_name]
        
        # Calculate deviations for all methods
        deviation_dominant = self._calculate_enhanced_color_deviation(dominant_color, target_rgb)
        deviation_avg = self._calculate_enhanced_color_deviation(avg_color, target_rgb)
        deviation_mode = self._calculate_enhanced_color_deviation(mode_color, target_rgb)
        deviation_hist = self._calculate_enhanced_color_deviation(hist_color, target_rgb)
        
        # Select best result using weighted scoring
        methods = [
            (dominant_color, deviation_dominant, "enhanced_kmeans", 0.3),
            (avg_color, deviation_avg, "weighted_average", 0.25),
            (mode_color, deviation_mode, "mode_analysis", 0.25),
            (hist_color, deviation_hist, "histogram_peak", 0.2)
        ]
        
        # Calculate weighted score for each method
        best_score = float('inf')
        final_color = dominant_color
        final_deviation = deviation_dominant
        method = "enhanced_kmeans"
        
        for color, deviation, method_name, weight in methods:
            # Penalize extreme deviations more heavily
            weighted_score = deviation * weight + (deviation ** 2) * 0.1
            if weighted_score < best_score:
                best_score = weighted_score
                final_color = color
                final_deviation = deviation
                method = method_name
        
        # Enhanced tolerance check with color-specific adjustments
        base_tolerance = 0.05  # 5% base tolerance
        color_specific_tolerance = self._get_color_specific_tolerance(color_name)
        final_tolerance = min(base_tolerance, color_specific_tolerance)
        
        status = 'pass' if final_deviation <= final_tolerance else 'fail'
        
        # Enhanced confidence calculation
        confidence = self._calculate_color_confidence(final_deviation, final_tolerance, methods)
        
        return {
            'status': status,
            'deviation': f"{final_deviation*100:.1f}%",
            'detected_rgb': final_color.tolist() if hasattr(final_color, 'tolist') else list(final_color),
            'expected_rgb': target_rgb,
            'method': method,
            'confidence': f"{confidence:.1f}%",
            'tolerance_used': f"{final_tolerance*100:.1f}%"
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
        deviation = self._calculate_enhanced_color_deviation(avg_blue, target_rgb)
        
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
    
    def _get_dominant_color_enhanced_kmeans(self, region):
        """Enhanced K-means clustering with multiple sampling strategies"""
        pixels = region.reshape(-1, 3).astype(np.float32)
        
        # Intelligent noise filtering based on color distribution
        mean_intensity = np.mean(pixels)
        std_intensity = np.std(pixels)
        
        # Dynamic filtering based on image characteristics
        lower_bound = max(5, mean_intensity - 2 * std_intensity)
        upper_bound = min(250, mean_intensity + 2 * std_intensity)
        
        # Remove noise pixels
        intensity_mask = np.all((pixels >= lower_bound) & (pixels <= upper_bound), axis=1)
        clean_pixels = pixels[intensity_mask]
        
        if len(clean_pixels) < 50:  # Fallback if too few pixels
            return np.mean(pixels, axis=0).astype(int)
        
        # Multiple K-means runs with different cluster counts
        best_result = None
        best_silhouette = -1
        
        for n_clusters in [2, 3, 4]:  # Try different cluster counts
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', max_iter=100)
                labels = kmeans.fit_predict(clean_pixels)
                
                # Calculate silhouette score for cluster quality
                if len(np.unique(labels)) > 1:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(clean_pixels, labels)
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        
                        # Get the largest cluster (most representative)
                        counts = np.bincount(labels)
                        dominant_cluster = np.argmax(counts)
                        best_result = kmeans.cluster_centers_[dominant_cluster]
                        
            except Exception:
                continue
        
        if best_result is not None:
            return best_result.astype(int)
        else:
            # Fallback to weighted average
            return self._get_weighted_average_color(region)
    
    def _get_weighted_average_color(self, region):
        """Get weighted average color with intelligent outlier removal"""
        pixels = region.reshape(-1, 3).astype(np.float32)
        
        # Calculate weights based on distance from median
        median_color = np.median(pixels, axis=0)
        distances = np.linalg.norm(pixels - median_color, axis=1)
        
        # Use inverse distance as weight (closer to median = higher weight)
        weights = 1.0 / (1.0 + distances)
        
        # Remove extreme outliers (beyond 3 standard deviations)
        distance_threshold = np.mean(distances) + 3 * np.std(distances)
        valid_mask = distances <= distance_threshold
        
        valid_pixels = pixels[valid_mask]
        valid_weights = weights[valid_mask]
        
        if len(valid_pixels) == 0:
            return np.mean(pixels, axis=0).astype(int)
        
        # Calculate weighted average
        weighted_avg = np.average(valid_pixels, axis=0, weights=valid_weights)
        
        return weighted_avg.astype(int)
    
    def _get_mode_color(self, region):
        """Get mode color using histogram analysis"""
        pixels = region.reshape(-1, 3)
        
        # Reduce color resolution for better mode detection
        quantized_pixels = (pixels // 8) * 8  # Quantize to 8-level steps
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(quantized_pixels, axis=0, return_counts=True)
        
        # Get the most frequent color
        mode_index = np.argmax(counts)
        mode_color = unique_colors[mode_index]
        
        return mode_color.astype(int)
    
    def _get_histogram_peak_color(self, region):
        """Get color using 3D histogram peak analysis"""
        pixels = region.reshape(-1, 3)
        
        # Create 3D histogram with reduced bins for efficiency
        hist, edges = np.histogramdd(pixels, bins=16, range=[(0, 256), (0, 256), (0, 256)])
        
        # Find the peak in the histogram
        peak_indices = np.unravel_index(np.argmax(hist), hist.shape)
        
        # Convert indices back to RGB values
        peak_color = []
        for i, edge_array in enumerate(edges):
            bin_center = (edge_array[peak_indices[i]] + edge_array[peak_indices[i] + 1]) / 2
            peak_color.append(int(bin_center))
        
        return np.array(peak_color)
    
    def _get_color_specific_tolerance(self, color_name):
        """Get color-specific tolerance values"""
        # Different colors have different perceptual tolerances
        tolerances = {
            'saffron': 0.045,   # Slightly tighter for saffron (orange is sensitive)
            'white': 0.03,      # Very tight for white (should be pure)
            'green': 0.05,      # Standard tolerance for green
            'chakra_blue': 0.055 # Slightly more lenient for navy blue (darker colors vary more)
        }
        return tolerances.get(color_name, 0.05)
    
    def _calculate_color_confidence(self, deviation, tolerance, methods):
        """Calculate enhanced confidence score"""
        # Base confidence from deviation
        base_confidence = max(0, 100 - (deviation / tolerance) * 100)
        
        # Boost confidence if multiple methods agree
        deviations = [method[1] for method in methods]
        deviation_std = np.std(deviations)
        
        # Lower standard deviation means methods agree more
        agreement_bonus = max(0, 20 - deviation_std * 1000)
        
        final_confidence = min(100, base_confidence + agreement_bonus)
        return final_confidence
    
    def _calculate_enhanced_color_deviation(self, color1, color2):
        """Enhanced color deviation calculation using multiple color spaces"""
        
        # Ensure colors are in the right format
        if hasattr(color1, 'tolist'):
            color1 = color1.astype(int)
        if hasattr(color2, 'tolist'):
            color2 = color2
        
        color1 = np.array(color1, dtype=np.uint8)
        color2 = np.array(color2, dtype=np.uint8)
        
        # Method 1: Delta-E in LAB space (perceptual accuracy)
        try:
            color1_bgr = np.array([[[color1[2], color1[1], color1[0]]]], dtype=np.uint8)
            color2_bgr = np.array([[[color2[2], color2[1], color2[0]]]], dtype=np.uint8)
            
            lab1 = cv2.cvtColor(color1_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
            lab2 = cv2.cvtColor(color2_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
            
            # Enhanced Delta-E calculation (CIE94)
            delta_l = lab1[0] - lab2[0]
            delta_a = lab1[1] - lab2[1]
            delta_b = lab1[2] - lab2[2]
            
            # CIE94 formula for better perceptual accuracy
            kl, kc, kh = 1.0, 1.0, 1.0  # Weighting factors
            sl = 1.0
            c1 = np.sqrt(lab1[1]**2 + lab1[2]**2)
            c2 = np.sqrt(lab2[1]**2 + lab2[2]**2)
            delta_c = c1 - c2
            delta_h_sq = delta_a**2 + delta_b**2 - delta_c**2
            delta_h = np.sqrt(max(0.0, float(delta_h_sq)))
            
            sc = 1 + 0.045 * c1
            sh = 1 + 0.015 * c1
            
            delta_e94 = np.sqrt(
                (delta_l / (kl * sl))**2 +
                (delta_c / (kc * sc))**2 +
                (delta_h / (kh * sh))**2
            )
            
            lab_deviation = min(delta_e94 / 100.0, 1.0)
        except Exception:
            # Fallback to simple Euclidean distance
            lab_deviation = np.linalg.norm(color1.astype(float) - color2.astype(float)) / (255 * np.sqrt(3))
        
        # Method 2: HSV space deviation (for hue accuracy)
        hsv_deviation = lab_deviation  # Default fallback
        try:
            if 'color1_bgr' in locals() and 'color2_bgr' in locals():
                hsv1 = cv2.cvtColor(color1_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
                hsv2 = cv2.cvtColor(color2_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
            else:
                # Recreate BGR arrays
                color1_bgr = np.array([[[color1[2], color1[1], color1[0]]]], dtype=np.uint8)
                color2_bgr = np.array([[[color2[2], color2[1], color2[0]]]], dtype=np.uint8)
                hsv1 = cv2.cvtColor(color1_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
                hsv2 = cv2.cvtColor(color2_bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
            
            # Handle hue wraparound (0-179 in OpenCV)
            hue_diff = abs(hsv1[0] - hsv2[0])
            if hue_diff > 90:
                hue_diff = 180 - hue_diff
            
            # Weighted HSV deviation
            hsv_deviation = (
                (hue_diff / 90.0) * 0.5 +  # Hue weight
                (abs(hsv1[1] - hsv2[1]) / 255.0) * 0.3 +  # Saturation weight
                (abs(hsv1[2] - hsv2[2]) / 255.0) * 0.2   # Value weight
            )
        except Exception:
            hsv_deviation = lab_deviation
        
        # Method 3: RGB space deviation (for digital accuracy)
        rgb_deviation = np.linalg.norm(color1.astype(float) - color2.astype(float)) / (255 * np.sqrt(3))
        
        # Combine deviations with weights
        final_deviation = (
            lab_deviation * 0.6 +    # Perceptual accuracy
            hsv_deviation * 0.25 +   # Hue accuracy
            rgb_deviation * 0.15     # Digital accuracy
        )
        
        return min(final_deviation, 1.0)
