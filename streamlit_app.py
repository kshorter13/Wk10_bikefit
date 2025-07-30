import streamlit as st
import cv2
import numpy as np
import tempfile

# Only import MediaPipe after OpenCV works
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

def simple_angle_calc(p1, p2, p3):
    """Basic angle calculation"""
    try:
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y]) 
        c = np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine, -1, 1))
        return np.degrees(angle)
    except:
        return None

def main():
    st.title("🦴 Simple Joint Angles")
    
    if not MEDIAPIPE_AVAILABLE:
        st.error("MediaPipe not available - check requirements.txt")
        return
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)
    
    uploaded_file = st.file_uploader("Upload Video", type=['mp4'])
    
    if uploaded_file:
        # Save file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        st.write(f"Frames: {frame_count}, FPS: {fps:.1f}")
        
        # Frame selector
        frame_num = st.slider("Frame", 0, frame_count-1, 0)
        
        # Get frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(rgb_frame, caption=f"Frame {frame_num}")
            
            with col2:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Simple knee angles only
                    try:
                        # Left knee
                        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value] 
                        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                        
                        left_angle = simple_angle_calc(left_hip, left_knee, left_ankle)
                        if left_angle:
                            st.metric("Left Knee", f"{left_angle:.1f}°")
                        
                        # Right knee  
                        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                        
                        right_angle = simple_angle_calc(right_hip, right_knee, right_ankle)
                        if right_angle:
                            st.metric("Right Knee", f"{right_angle:.1f}°")
                            
                    except:
                        st.warning("Could not calculate angles")
                else:
                    st.warning("No pose detected")
    else:
        st.info("Upload a video to start analysis")

if __name__ == "__main__":
    main()
