import streamlit as st
import json
import time
from flag_validator import FlagValidator
from PIL import Image
import io

def main():
    st.set_page_config(
        page_title="🇮🇳 Indian Flag Validator - Championship Grade",
        page_icon="🇮🇳",
        layout="wide"
    )
    
    st.title("🇮🇳 Indian Flag Image Validator")
    st.markdown("### Championship-Grade BIS Compliance Checker with Sub-Pixel Precision")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    debug_mode = st.sidebar.checkbox("Enable Visual Debug Mode", value=False)
    competition_mode = st.sidebar.checkbox("Competition Mode (High Accuracy)", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Flag Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'svg'],
            help="Upload PNG, JPG, or SVG image (max 5MB)"
        )
        
        if uploaded_file is not None:
            # Display image info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📁 File: {uploaded_file.name} ({file_size/1024/1024:.2f} MB)")
            
            # Show uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Validate button
                if st.button("🔍 Validate Flag", type="primary"):
                    validate_flag(uploaded_file, debug_mode, competition_mode, show_confidence)
                    
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
    
    with col2:
        st.header("📋 BIS Specifications")
        st.markdown("""
        **Aspect Ratio:** 3:2 (±0.1% tolerance in competition mode)
        
        **Colors (±5% RGB tolerance):**
        - 🟠 Saffron: #FF9933
        - ⚪ White: #FFFFFF  
        - 🟢 Green: #138808
        - 🔵 Navy Blue (Chakra): #000080
        
        **Stripe Proportions:** Each band = 1/3 of height
        
        **Ashoka Chakra:**
        - Diameter: 3/4 of white band height
        - Exactly 24 evenly spaced spokes
        - Perfectly centered in white band
        """)

def validate_flag(uploaded_file, debug_mode=False, competition_mode=True, show_confidence=True):
    """Main validation function"""
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Start timing
        start_time = time.time()
        
        # Initialize validator
        status_text.text("🔧 Initializing validator...")
        progress_bar.progress(10)
        
        validator = FlagValidator(
            debug_mode=debug_mode,
            competition_mode=competition_mode
        )
        
        # Load and preprocess image
        status_text.text("📷 Loading and preprocessing image...")
        progress_bar.progress(30)
        
        # Convert uploaded file to bytes
        image_bytes = uploaded_file.getvalue()
        
        # Validate the flag
        status_text.text("🔍 Running validation algorithms...")
        progress_bar.progress(50)
        
        result = validator.validate_flag(image_bytes)
        
        # Complete progress
        progress_bar.progress(100)
        processing_time = time.time() - start_time
        
        # Display results
        display_results(result, processing_time, show_confidence, debug_mode, validator)
        
    except Exception as e:
        st.error(f"❌ Validation failed: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_results(result, processing_time, show_confidence, debug_mode, validator):
    """Display validation results"""
    
    st.success(f"✅ Validation completed in {processing_time:.2f} seconds")
    
    # Overall status
    overall_pass = all([
        result['aspect_ratio']['status'] == 'pass',
        all(color['status'] == 'pass' for color in result['colors'].values()),
        result['stripe_proportion']['status'] == 'pass',
        result['chakra_position']['status'] == 'pass',
        result['chakra_spokes']['status'] == 'pass'
    ])
    
    if overall_pass:
        st.balloons()
        st.success("🎉 **FLAG PASSES ALL BIS SPECIFICATIONS!**")
    else:
        st.warning("⚠️ **FLAG HAS COMPLIANCE ISSUES**")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Validation Report", "🔧 Raw JSON", "🖼️ Debug Visuals"])
    
    with tab1:
        display_formatted_results(result, show_confidence)
    
    with tab2:
        st.code(json.dumps(result, indent=2), language='json')
    
    with tab3:
        if debug_mode and hasattr(validator, 'debug_image') and validator.debug_image is not None:
            st.image(validator.debug_image, caption="Debug Visualization", use_column_width=True)
        else:
            st.info("Enable 'Visual Debug Mode' to see detection visualizations")

def display_formatted_results(result, show_confidence):
    """Display formatted validation results"""
    
    # Aspect Ratio
    st.subheader("📐 Aspect Ratio")
    aspect_status = result['aspect_ratio']['status']
    if aspect_status == 'pass':
        st.success(f"✅ **PASS** - Actual ratio: {result['aspect_ratio']['actual']}")
    else:
        st.error(f"❌ **FAIL** - Actual ratio: {result['aspect_ratio']['actual']} (Expected: 1.5)")
    
    if show_confidence and 'confidence' in result['aspect_ratio']:
        st.caption(f"Confidence: {result['aspect_ratio']['confidence']}")
    
    # Colors
    st.subheader("🎨 Color Accuracy")
    color_names = {
        'saffron': '🟠 Saffron',
        'white': '⚪ White', 
        'green': '🟢 Green',
        'chakra_blue': '🔵 Chakra Blue'
    }
    
    for color_key, color_data in result['colors'].items():
        color_name = color_names.get(color_key, color_key)
        if color_data['status'] == 'pass':
            st.success(f"✅ **{color_name}** - Deviation: {color_data['deviation']}")
        else:
            st.error(f"❌ **{color_name}** - Deviation: {color_data['deviation']} (>5%)")
        
        if show_confidence and 'confidence' in color_data:
            st.caption(f"Confidence: {color_data['confidence']}")
    
    # Stripe Proportions
    st.subheader("📏 Stripe Proportions")
    stripe_status = result['stripe_proportion']['status']
    if stripe_status == 'pass':
        st.success(f"✅ **PASS** - Top: {result['stripe_proportion']['top']}, Middle: {result['stripe_proportion']['middle']}, Bottom: {result['stripe_proportion']['bottom']}")
    else:
        st.error(f"❌ **FAIL** - Top: {result['stripe_proportion']['top']}, Middle: {result['stripe_proportion']['middle']}, Bottom: {result['stripe_proportion']['bottom']}")
    
    # Chakra Position
    st.subheader("🎯 Chakra Position")
    chakra_pos_status = result['chakra_position']['status']
    if chakra_pos_status == 'pass':
        st.success(f"✅ **CENTERED** - Offset: X={result['chakra_position']['offset_x']}, Y={result['chakra_position']['offset_y']}")
    else:
        st.error(f"❌ **OFF-CENTER** - Offset: X={result['chakra_position']['offset_x']}, Y={result['chakra_position']['offset_y']}")
    
    # Chakra Spokes
    st.subheader("⚙️ Chakra Spokes")
    spokes_status = result['chakra_spokes']['status']
    detected_spokes = result['chakra_spokes']['detected']
    if spokes_status == 'pass':
        st.success(f"✅ **CORRECT** - Detected: {detected_spokes} spokes")
    else:
        st.error(f"❌ **INCORRECT** - Detected: {detected_spokes} spokes (Expected: 24)")
    
    if show_confidence and 'confidence' in result['chakra_spokes']:
        st.caption(f"Confidence: {result['chakra_spokes']['confidence']}")

if __name__ == "__main__":
    main()
