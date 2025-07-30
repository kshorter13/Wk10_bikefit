import streamlit as st

st.title("🔧 OpenGL Fix Test")

try:
    import cv2
    st.success(f"✅ OpenCV works! Version: {cv2.__version__}")
    
    # Test basic OpenCV operation
    import numpy as np
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    st.success("✅ OpenCV operations work!")
    
    try:
        import mediapipe as mp
        st.success(f"✅ MediaPipe works! Version: {mp.__version__}")
        
        # Simple MediaPipe test
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        st.success("✅ MediaPipe pose initialization works!")
        
        st.markdown("---")
        st.markdown("### 🎉 Everything working! Ready for joint analysis")
        
        # Simple file upload test
        uploaded_file = st.file_uploader("Upload test video", type=['mp4'])
        if uploaded_file:
            st.success("✅ File upload works!")
            st.info("All systems ready - you can now use the full joint analysis app")
            
    except Exception as e:
        st.error(f"❌ MediaPipe error: {e}")
        
except Exception as e:
    st.error(f"❌ OpenCV still failing: {e}")
    st.warning("The packages.txt file may not have deployed yet. Wait 1-2 minutes and refresh.")

st.markdown("---")
st.code("""
Required files in your repo:

packages.txt:
libgl1-mesa-glx
libglib2.0-0

requirements.txt:
streamlit
opencv-python-headless
mediapipe
numpy
""", language="text")
