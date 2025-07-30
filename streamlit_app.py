import streamlit as st

st.title("🔧 OpenCV Installation Test")

# Test 1: Basic imports
try:
    import numpy as np
    st.success("✅ NumPy imported successfully")
    st.write(f"NumPy version: {np.__version__}")
except Exception as e:
    st.error(f"❌ NumPy failed: {e}")

# Test 2: OpenCV import
try:
    import cv2
    st.success("✅ OpenCV imported successfully!")
    st.write(f"OpenCV version: {cv2.__version__}")
    
    # Test 3: MediaPipe import
    try:
        import mediapipe as mp
        st.success("✅ MediaPipe imported successfully!")
        st.write(f"MediaPipe version: {mp.__version__}")
        
        # If all imports work, show the main app
        st.markdown("---")
        st.markdown("### 🎉 All imports successful! Ready for joint angle analysis.")
        
        # Simple file uploader test
        uploaded_file = st.file_uploader("Test file upload", type=['mp4'])
        if uploaded_file:
            st.success("File upload works!")
            
    except Exception as e:
        st.error(f"❌ MediaPipe failed: {e}")
        
except Exception as e:
    st.error(f"❌ OpenCV import failed: {e}")
    st.warning("This is the main issue - OpenCV is not installing properly")
    
    st.markdown("### 🔧 Try these requirements.txt versions:")
    st.code("""
# Option 1:
opencv-python-headless==4.8.1.78
streamlit
numpy

# Option 2: 
opencv-contrib-python-headless==4.10.0.84
streamlit
numpy

# Option 3:
opencv-python-headless
streamlit
numpy
    """)

st.markdown("---")
st.info("💡 Replace your streamlit_app.py with this test first, then try the requirements above")
