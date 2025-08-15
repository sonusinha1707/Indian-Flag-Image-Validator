import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """Advanced image preprocessing and enhancement pipeline"""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
    
    def load_image(self, image_bytes):
        """Load image from bytes with error handling"""
        try:
            # Try PIL first
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format (BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
            
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def preprocess_image(self, image):
        """Multi-stage preprocessing pipeline"""
        
        # Stage 1: Noise reduction
        denoised = self._denoise_image(image)
        
        # Stage 2: Contrast enhancement
        enhanced = self._enhance_contrast(denoised)
        
        # Stage 3: Edge enhancement for better detection
        sharpened = self._sharpen_image(enhanced)
        
        # Stage 4: Color normalization
        normalized = self._normalize_colors(sharpened)
        
        return normalized
    
    def _denoise_image(self, image):
        """Remove noise while preserving edges"""
        # Use Non-local Means Denoising
        if len(image.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Grayscale image
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def _enhance_contrast(self, image):
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _sharpen_image(self, image):
        """Apply unsharp masking for edge enhancement"""
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask
    
    def _normalize_colors(self, image):
        """Normalize colors for consistent analysis"""
        # Convert to float for processing
        normalized = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        gamma = 1.2
        normalized = np.power(normalized, 1.0 / gamma)
        
        # White balance correction
        normalized = self._white_balance(normalized)
        
        # Convert back to uint8
        normalized = (normalized * 255).astype(np.uint8)
        
        return normalized
    
    def _white_balance(self, image):
        """Simple white balance correction"""
        # Calculate average values for each channel
        avg_b = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_r = np.mean(image[:, :, 2])
        
        # Calculate scaling factors
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        scale_b = avg_gray / avg_b if avg_b > 0 else 1
        scale_g = avg_gray / avg_g if avg_g > 0 else 1
        scale_r = avg_gray / avg_r if avg_r > 0 else 1
        
        # Apply scaling (with limits to prevent overcorrection)
        scale_b = np.clip(scale_b, 0.5, 2.0)
        scale_g = np.clip(scale_g, 0.5, 2.0)
        scale_r = np.clip(scale_r, 0.5, 2.0)
        
        balanced = image.copy()
        balanced[:, :, 0] *= scale_b
        balanced[:, :, 1] *= scale_g
        balanced[:, :, 2] *= scale_r
        
        # Clip values to valid range
        balanced = np.clip(balanced, 0, 1)
        
        return balanced
    
    def resize_if_needed(self, image, max_dimension=2048):
        """Resize image if it's too large for processing"""
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            # Calculate scaling factor
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize using high-quality interpolation
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            return resized
        
        return image
