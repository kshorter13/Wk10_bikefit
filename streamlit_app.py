import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

class JointAngleAnalyzer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)"""
        try:
            # Convert to numpy arrays
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except:
            return None
    
    def get_all_joint_angles(self, landmarks):
        """Calculate all joint angles for both sides"""
        angles = {
            'left_ankle': None, 'left_knee': None, 'left_hip': None, 'left_elbow': None,
            'right_ankle': None, 'right_knee': None, 'right_hip': None, 'right_elbow': None
        }
        
        try:
            # Left side landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
            
            # Right side landmarks
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
            
            # Calculate LEFT side angles
            # Left elbow (shoulder-elbow-wrist)
            if all(p.visibility > 0.5 for p in [left_shoulder, left_elbow, left_wrist]):
                angles['left_elbow'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Left hip (shoulder-hip-knee)
            if all(p.visibility > 0.5 for p in [left_shoulder, left_hip, left_knee]):
                angles['left_hip'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
            
            # Left knee (hip-knee-ankle)
            if all(p.visibility > 0.5 for p in [left_hip, left_knee, left_ankle]):
                angles['left_knee'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Left ankle (knee-ankle-foot)
            if all(p.visibility > 0.5 for p in [left_knee, left_ankle, left_foot_index]):
                angles['left_ankle'] = self.calculate_angle(left_knee, left_ankle, left_foot_index)
            
            # Calculate RIGHT side angles
            # Right elbow (shoulder-elbow-wrist)
            if all(p.visibility > 0.5 for p in [right_shoulder, right_elbow, right_wrist]):
                angles['right_elbow'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Right hip (shoulder-hip-knee)
            if all(p.visibility > 0.5 for p in [right_shoulder, right_hip, right_knee]):
                angles['right_hip'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Right knee (hip-knee-ankle)
            if all(p.visibility > 0.5 for p in [right_hip, right_knee, right_ankle]):
                angles['right_knee'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Right ankle (knee-ankle-foot)
            if all(p.visibility > 0.5 for p in [right_knee, right_ankle, right_foot_index]):
                angles['right_ankle'] = self.calculate_angle(right_knee, right_ankle, right_foot_index)
                
        except Exception as e:
            pass
        
        return angles
    
    def process_frame(self, frame):
        """Process frame and return annotated frame with joint angles"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        angles = {
            'left_ankle': None, 'left_knee': None, 'left_hip': None, 'left_elbow': None,
            'right_ankle': None, 'right_knee': None, 'right_hip': None, 'right_elbow': None
        }
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate joint angles
            angles = self.get_all_joint_angles(results.pose_landmarks.landmark)
        
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), angles
    
    def get_video_info(self, video_path):
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frame_count, fps, width, height
    
    def get_frame_at_position(self, video_path, frame_number):
        """Get specific frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

def main():
    st.set_page_config(
        page_title="Joint Angle Analysis",
        page_icon="🦴",
        layout="wide"
    )
    
    st.title("🦴 Complete Joint Angle Analysis")
    st.markdown("*Analyze ankle, knee, hip, and elbow angles for both sides using MediaPipe*")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = JointAngleAnalyzer()
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'current_angles' not in st.session_state:
        st.session_state.current_angles = {}
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to analyze joint angles"
    )
    
    if uploaded_file is not None:
        # Save file temporarily
        if not st.session_state.video_loaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_loaded = True
        
        try:
            # Get video information
            frame_count, fps, width, height = st.session_state.analyzer.get_video_info(st.session_state.video_path)
            
            # Display video information
            st.subheader("📹 Video Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("FPS", f"{fps:.1f}")
            with col2:
                st.metric("Total Frames", frame_count)
            with col3:
                st.metric("Duration", f"{frame_count/fps:.2f}s")
            with col4:
                st.metric("Resolution", f"{width}x{height}")
            
            # Frame navigation
            st.subheader("🎬 Frame Navigation")
            
            # Current frame selector
            current_frame = st.slider(
                "Current Frame",
                min_value=0,
                max_value=frame_count-1,
                value=0,
                help="Navigate through video frames"
            )
            
            # Navigation controls
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                if st.button("⏮️ Start"):
                    st.session_state.current_frame = 0
                    st.rerun()
            
            with col2:
                if st.button("⏪ -10"):
                    st.session_state.current_frame = max(0, current_frame - 10)
                    st.rerun()
            
            with col3:
                if st.button("⏪ -1"):
                    st.session_state.current_frame = max(0, current_frame - 1)
                    st.rerun()
            
            with col4:
                if st.button("⏩ +1"):
                    st.session_state.current_frame = min(frame_count-1, current_frame + 1)
                    st.rerun()
            
            with col5:
                if st.button("⏩ +10"):
                    st.session_state.current_frame = min(frame_count-1, current_frame + 10)
                    st.rerun()
            
            with col6:
                if st.button("⏭️ End"):
                    st.session_state.current_frame = frame_count-1
                    st.rerun()
            
            # Process current frame
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.spinner("Processing frame..."):
                    frame = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, current_frame)
                    
                    if frame is not None:
                        annotated_frame, angles = st.session_state.analyzer.process_frame(frame)
                        st.session_state.current_angles = angles
                        
                        current_time = current_frame / fps
                        st.image(
                            annotated_frame,
                            caption=f"Frame {current_frame} | Time: {current_time:.3f}s | Pose landmarks with joint angles",
                            use_column_width=True
                        )
                    else:
                        st.error("Could not load frame")
            
            with col2:
                st.subheader("📐 Joint Angles")
                
                # Current frame info
                current_time = current_frame / fps
                st.metric("Current Time", f"{current_time:.3f}s")
                st.metric("Frame Number", current_frame)
                
                # Display angles in organized layout
                if st.session_state.current_angles:
                    angles = st.session_state.current_angles
                    
                    # Left side
                    st.markdown("### 👈 Left Side")
                    col_l1, col_l2 = st.columns(2)
                    
                    with col_l1:
                        if angles['left_elbow'] is not None:
                            st.metric("🦾 Elbow", f"{angles['left_elbow']:.1f}°")
                        else:
                            st.metric("🦾 Elbow", "N/A")
                        
                        if angles['left_knee'] is not None:
                            st.metric("🦵 Knee", f"{angles['left_knee']:.1f}°")
                        else:
                            st.metric("🦵 Knee", "N/A")
                    
                    with col_l2:
                        if angles['left_hip'] is not None:
                            st.metric("🦴 Hip", f"{angles['left_hip']:.1f}°")
                        else:
                            st.metric("🦴 Hip", "N/A")
                        
                        if angles['left_ankle'] is not None:
                            st.metric("🦶 Ankle", f"{angles['left_ankle']:.1f}°")
                        else:
                            st.metric("🦶 Ankle", "N/A")
                    
                    # Right side
                    st.markdown("### 👉 Right Side")
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        if angles['right_elbow'] is not None:
                            st.metric("🦾 Elbow", f"{angles['right_elbow']:.1f}°")
                        else:
                            st.metric("🦾 Elbow", "N/A")
                        
                        if angles['right_knee'] is not None:
                            st.metric("🦵 Knee", f"{angles['right_knee']:.1f}°")
                        else:
                            st.metric("🦵 Knee", "N/A")
                    
                    with col_r2:
                        if angles['right_hip'] is not None:
                            st.metric("🦴 Hip", f"{angles['right_hip']:.1f}°")
                        else:
                            st.metric("🦴 Hip", "N/A")
                        
                        if angles['right_ankle'] is not None:
                            st.metric("🦶 Ankle", f"{angles['right_ankle']:.1f}°")
                        else:
                            st.metric("🦶 Ankle", "N/A")
                
                # Angle reference guide
                with st.expander("📊 Angle Reference"):
                    st.markdown("""
                    **Angle Definitions:**
                    - **🦾 Elbow**: Shoulder-Elbow-Wrist
                    - **🦴 Hip**: Shoulder-Hip-Knee  
                    - **🦵 Knee**: Hip-Knee-Ankle
                    - **🦶 Ankle**: Knee-Ankle-Foot
                    
                    **Interpretation:**
                    - **~180°**: Fully extended/straight
                    - **~90°**: Right angle bend
                    - **<90°**: Deep flexion
                    - **>90°**: Partial extension
                    """)
                
                # Data export
                if st.button("📊 Export Current Angles", use_container_width=True):
                    if st.session_state.current_angles:
                        angles = st.session_state.current_angles
                        export_text = f"""Frame {current_frame} Angle Data:
Time: {current_time:.3f}s

LEFT SIDE:
Elbow: {angles['left_elbow']:.1f}° if angles['left_elbow'] else 'N/A'
Hip: {angles['left_hip']:.1f}° if angles['left_hip'] else 'N/A'
Knee: {angles['left_knee']:.1f}° if angles['left_knee'] else 'N/A'
Ankle: {angles['left_ankle']:.1f}° if angles['left_ankle'] else 'N/A'

RIGHT SIDE:
Elbow: {angles['right_elbow']:.1f}° if angles['right_elbow'] else 'N/A'
Hip: {angles['right_hip']:.1f}° if angles['right_hip'] else 'N/A'
Knee: {angles['right_knee']:.1f}° if angles['right_knee'] else 'N/A'
Ankle: {angles['right_ankle']:.1f}° if angles['right_ankle'] else 'N/A'

Video: {uploaded_file.name}"""
                        
                        st.code(export_text, language="text")
                    else:
                        st.warning("No angle data available")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("Make sure the video file is valid and contains visible human poses")
    
    else:
        # Instructions when no video loaded
        st.markdown("""
        ### 📋 How to Use:
        
        1. **📤 Upload Video**: Choose a video file with clear body movements
        2. **🎬 Navigate**: Use slider and controls to move through frames  
        3. **📐 View Angles**: See all 8 joint angles updated in real-time
        4. **📊 Export**: Copy current frame angle data
        
        ### 🎯 Tracked Joints (Both Sides):
        - **🦾 Elbow**: Shoulder-Elbow-Wrist angle
        - **🦴 Hip**: Shoulder-Hip-Knee angle
        - **🦵 Knee**: Hip-Knee-Ankle angle  
        - **🦶 Ankle**: Knee-Ankle-Foot angle
        
        ### 💡 Best Results:
        - **Full body visible** in the frame
        - **Good lighting** and contrast
        - **Clear view** of all joints
        - **Front or side view** works best
        - **Person should be the main subject**
        
        ### 📊 Applications:
        - **Exercise form analysis**
        - **Gait analysis and rehabilitation**
        - **Sports biomechanics**
        - **Physical therapy assessment**
        - **Movement pattern research**
        
        ### ⚡ Performance Tips:
        - **Smaller video files** process faster
        - **720p resolution** is usually sufficient
        - **Good pose visibility** improves accuracy
        """)

if __name__ == "__main__":
    main()
