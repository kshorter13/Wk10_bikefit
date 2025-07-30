import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

class LightweightJointAnalyzer:
    def __init__(self):
        # Initialize MediaPipe with minimal settings
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Lightweight pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Process each frame independently
            model_complexity=0,      # Lightest model
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    
    def calculate_angle(self, p1, p2, p3):
        """Simple angle calculation"""
        try:
            a = np.array([p1.x, p1.y])
            b = np.array([p2.x, p2.y])
            c = np.array([p3.x, p3.y])
            
            ba = a - b
            bc = c - b
            
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.arccos(cosine)
            
            return np.degrees(angle)
        except:
            return None
    
    def get_key_angles(self, landmarks):
        """Get only the most important angles"""
        angles = {}
        
        try:
            # Key landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Calculate key angles only
            if all(p.visibility > 0.3 for p in [left_shoulder, left_elbow, left_wrist]):
                angles['left_elbow'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if all(p.visibility > 0.3 for p in [right_shoulder, right_elbow, right_wrist]):
                angles['right_elbow'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if all(p.visibility > 0.3 for p in [left_hip, left_knee, left_ankle]):
                angles['left_knee'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            if all(p.visibility > 0.3 for p in [right_hip, right_knee, right_ankle]):
                angles['right_knee'] = self.calculate_angle(right_hip, right_knee, right_ankle)
                
        except:
            pass
        
        return angles
    
    def process_frame_simple(self, frame):
        """Lightweight frame processing"""
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        angles = {}
        pose_detected = False
        
        if results.pose_landmarks:
            pose_detected = True
            angles = self.get_key_angles(results.pose_landmarks.landmark)
            
            # Draw simple pose lines
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), angles, pose_detected
    
    def get_video_info(self, video_path):
        """Get basic video info"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frame_count, fps
    
    def get_frame(self, video_path, frame_number):
        """Get single frame"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

def main():
    st.set_page_config(
        page_title="Joint Angle Analyzer",
        page_icon="🦴",
        layout="centered"
    )
    
    st.title("🦴 Joint Angle Analyzer")
    st.markdown("*Lightweight version - Elbow & Knee angles only*")
    
    # Initialize
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LightweightJointAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=['mp4', 'avi', 'mov'],
        help="Smaller videos work better"
    )
    
    if uploaded_file is not None:
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Get video info
            frame_count, fps = st.session_state.analyzer.get_video_info(video_path)
            
            # Basic info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Duration", f"{frame_count/fps:.1f}s")
            
            # Frame selector
            st.subheader("Frame Navigation")
            current_frame = st.slider(
                "Frame",
                0, frame_count-1, 0,
                help="Move slider to navigate video"
            )
            
            # Quick controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("⏮️ Start"):
                    current_frame = 0
                    st.rerun()
            with col2:
                if st.button("⏪ -10"):
                    current_frame = max(0, current_frame - 10)
                    st.rerun()
            with col3:
                if st.button("⏩ +10"):
                    current_frame = min(frame_count-1, current_frame + 10)
                    st.rerun()
            with col4:
                if st.button("⏭️ End"):
                    current_frame = frame_count-1
                    st.rerun()
            
            # Process current frame
            with st.spinner("Processing..."):
                frame = st.session_state.analyzer.get_frame(video_path, current_frame)
                
                if frame is not None:
                    processed_frame, angles, pose_detected = st.session_state.analyzer.process_frame_simple(frame)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.image(
                            processed_frame,
                            caption=f"Frame {current_frame} | {current_frame/fps:.2f}s",
                            use_column_width=True
                        )
                    
                    with col2:
                        st.subheader("Joint Angles")
                        
                        if pose_detected:
                            st.success("✅ Pose detected")
                            
                            # Left side
                            st.markdown("**Left Side:**")
                            if 'left_elbow' in angles:
                                st.write(f"Elbow: {angles['left_elbow']:.1f}°")
                            else:
                                st.write("Elbow: Not detected")
                                
                            if 'left_knee' in angles:
                                st.write(f"Knee: {angles['left_knee']:.1f}°")
                            else:
                                st.write("Knee: Not detected")
                            
                            # Right side
                            st.markdown("**Right Side:**")
                            if 'right_elbow' in angles:
                                st.write(f"Elbow: {angles['right_elbow']:.1f}°")
                            else:
                                st.write("Elbow: Not detected")
                                
                            if 'right_knee' in angles:
                                st.write(f"Knee: {angles['right_knee']:.1f}°")
                            else:
                                st.write("Knee: Not detected")
                        else:
                            st.warning("⚠️ No pose detected")
                            st.info("Try adjusting to a frame with clear body visibility")
                
                else:
                    st.error("Could not load frame")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        finally:
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass
    
    else:
        st.markdown("""
        ### 📋 Instructions:
        1. **Upload a video** (MP4, AVI, MOV)
        2. **Use the slider** to navigate frames
        3. **View joint angles** for elbow and knee
        4. **Clear body pose** works best
        
        ### 🎯 Features:
        - **Lightweight processing** for Streamlit Cloud
        - **Key joint angles**: Elbow & Knee only
        - **Real-time analysis** as you navigate
        - **Optimized for performance**
        
        ### 💡 Tips:
        - **Smaller video files** process faster
        - **Clear lighting** improves detection
        - **Front or side view** works best
        """)

if __name__ == "__main__":
    main()
