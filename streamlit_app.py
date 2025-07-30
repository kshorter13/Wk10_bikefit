import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

class FullBodyAngleAnalyzer:
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
        # Convert to numpy arrays
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def get_all_joint_angles(self, landmarks):
        """Calculate all joint angles for both sides"""
        angles = {
            'left_ankle': None,
            'left_knee': None,
            'left_hip': None,
            'left_elbow': None,
            'right_ankle': None,
            'right_knee': None,
            'right_hip': None,
            'right_elbow': None
        }
        
        try:
            # Left side landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value]
            left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
            
            # Right side landmarks
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value]
            right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
            
            # Calculate LEFT side angles
            # Left elbow angle (shoulder-elbow-wrist)
            if (left_shoulder.visibility > 0.5 and left_elbow.visibility > 0.5 and left_wrist.visibility > 0.5):
                angles['left_elbow'] = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Left hip angle (shoulder-hip-knee)
            if (left_shoulder.visibility > 0.5 and left_hip.visibility > 0.5 and left_knee.visibility > 0.5):
                angles['left_hip'] = self.calculate_angle(left_shoulder, left_hip, left_knee)
            
            # Left knee angle (hip-knee-ankle)
            if (left_hip.visibility > 0.5 and left_knee.visibility > 0.5 and left_ankle.visibility > 0.5):
                angles['left_knee'] = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Left ankle angle (knee-ankle-foot)
            if (left_knee.visibility > 0.5 and left_ankle.visibility > 0.5 and left_foot_index.visibility > 0.5):
                angles['left_ankle'] = self.calculate_angle(left_knee, left_ankle, left_foot_index)
            
            # Calculate RIGHT side angles
            # Right elbow angle (shoulder-elbow-wrist)
            if (right_shoulder.visibility > 0.5 and right_elbow.visibility > 0.5 and right_wrist.visibility > 0.5):
                angles['right_elbow'] = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Right hip angle (shoulder-hip-knee)
            if (right_shoulder.visibility > 0.5 and right_hip.visibility > 0.5 and right_knee.visibility > 0.5):
                angles['right_hip'] = self.calculate_angle(right_shoulder, right_hip, right_knee)
            
            # Right knee angle (hip-knee-ankle)
            if (right_hip.visibility > 0.5 and right_knee.visibility > 0.5 and right_ankle.visibility > 0.5):
                angles['right_knee'] = self.calculate_angle(right_hip, right_knee, right_ankle)
            
            # Right ankle angle (knee-ankle-foot)
            if (right_knee.visibility > 0.5 and right_ankle.visibility > 0.5 and right_foot_index.visibility > 0.5):
                angles['right_ankle'] = self.calculate_angle(right_knee, right_ankle, right_foot_index)
                
        except Exception as e:
            pass
        
        return angles
    
    def process_frame(self, frame):
        """Process frame and return annotated frame with all joint angles"""
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
            
            # Calculate all joint angles
            angles = self.get_all_joint_angles(results.pose_landmarks.landmark)
            
            # Draw angle text on frame
            y_offset = 30
            
            # Left side angles
            cv2.putText(annotated_frame, "LEFT SIDE:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            for joint in ['elbow', 'hip', 'knee', 'ankle']:
                angle_value = angles[f'left_{joint}']
                if angle_value is not None:
                    text = f'{joint.capitalize()}: {angle_value:.1f}°'
                    color = (0, 255, 0)  # Green
                else:
                    text = f'{joint.capitalize()}: N/A'
                    color = (0, 0, 255)  # Red
                
                cv2.putText(annotated_frame, text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 20
            
            # Right side angles
            y_offset += 10
            cv2.putText(annotated_frame, "RIGHT SIDE:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
            
            for joint in ['elbow', 'hip', 'knee', 'ankle']:
                angle_value = angles[f'right_{joint}']
                if angle_value is not None:
                    text = f'{joint.capitalize()}: {angle_value:.1f}°'
                    color = (0, 255, 0)  # Green
                else:
                    text = f'{joint.capitalize()}: N/A'
                    color = (0, 0, 255)  # Red
                
                cv2.putText(annotated_frame, text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 20
        
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
        page_title="Full Body Joint Analysis",
        page_icon="🦴",
        layout="wide"
    )
    
    st.title("🦴 Full Body Joint Angle Analysis")
    st.markdown("*Analyze ankle, knee, hip, and elbow angles for both sides using MediaPipe*")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FullBodyAngleAnalyzer()
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'angle_data' not in st.session_state:
        st.session_state.angle_data = {}
    
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
                    current_frame = 0
                    st.rerun()
            
            with col2:
                if st.button("⏪ -10"):
                    current_frame = max(0, current_frame - 10)
                    st.rerun()
            
            with col3:
                if st.button("⏪ -1"):
                    current_frame = max(0, current_frame - 1)
                    st.rerun()
            
            with col4:
                if st.button("⏩ +1"):
                    current_frame = min(frame_count-1, current_frame + 1)
                    st.rerun()
            
            with col5:
                if st.button("⏩ +10"):
                    current_frame = min(frame_count-1, current_frame + 10)
                    st.rerun()
            
            with col6:
                if st.button("⏭️ End"):
                    current_frame = frame_count-1
                    st.rerun()
            
            # Process current frame
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Check if frame is already processed
                if current_frame not in st.session_state.angle_data:
                    with st.spinner("Processing frame..."):
                        frame = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, current_frame)
                        
                        if frame is not None:
                            annotated_frame, angles = st.session_state.analyzer.process_frame(frame)
                            
                            # Store processed data
                            st.session_state.angle_data[current_frame] = {
                                'annotated_frame': annotated_frame,
                                'angles': angles
                            }
                
                # Display frame
                if current_frame in st.session_state.angle_data:
                    frame_data = st.session_state.angle_data[current_frame]
                    current_time = current_frame / fps
                    
                    st.image(
                        frame_data['annotated_frame'],
                        caption=f"Frame {current_frame} | Time: {current_time:.3f}s | All Joint Angles Displayed",
                        use_column_width=True
                    )
            
            with col2:
                st.subheader("📐 Joint Angles")
                
                # Current frame info
                current_time = current_frame / fps
                st.metric("Current Time", f"{current_time:.3f}s")
                st.metric("Frame Number", current_frame)
                
                # Display angles if available
                if current_frame in st.session_state.angle_data:
                    frame_data = st.session_state.angle_data[current_frame]
                    angles = frame_data['angles']
                    
                    # Left side angles
                    st.markdown("### 👈 Left Side")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Left elbow
                        if angles['left_elbow'] is not None:
                            st.metric("Elbow", f"{angles['left_elbow']:.1f}°")
                        else:
                            st.warning("Elbow: N/A")
                        
                        # Left knee
                        if angles['left_knee'] is not None:
                            st.metric("Knee", f"{angles['left_knee']:.1f}°")
                        else:
                            st.warning("Knee: N/A")
                    
                    with col2:
                        # Left hip
                        if angles['left_hip'] is not None:
                            st.metric("Hip", f"{angles['left_hip']:.1f}°")
                        else:
                            st.warning("Hip: N/A")
                        
                        # Left ankle
                        if angles['left_ankle'] is not None:
                            st.metric("Ankle", f"{angles['left_ankle']:.1f}°")
                        else:
                            st.warning("Ankle: N/A")
                    
                    # Right side angles
                    st.markdown("### 👉 Right Side")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Right elbow
                        if angles['right_elbow'] is not None:
                            st.metric("Elbow", f"{angles['right_elbow']:.1f}°")
                        else:
                            st.warning("Elbow: N/A")
                        
                        # Right knee
                        if angles['right_knee'] is not None:
                            st.metric("Knee", f"{angles['right_knee']:.1f}°")
                        else:
                            st.warning("Knee: N/A")
                    
                    with col2:
                        # Right hip
                        if angles['right_hip'] is not None:
                            st.metric("Hip", f"{angles['right_hip']:.1f}°")
                        else:
                            st.warning("Hip: N/A")
                        
                        # Right ankle
                        if angles['right_ankle'] is not None:
                            st.metric("Ankle", f"{angles['right_ankle']:.1f}°")
                        else:
                            st.warning("Ankle: N/A")
                
                # Joint angle reference
                with st.expander("📊 Angle Reference"):
                    st.markdown("""
                    **Joint Angle Definitions:**
                    - **Elbow**: Shoulder-Elbow-Wrist
                    - **Hip**: Shoulder-Hip-Knee  
                    - **Knee**: Hip-Knee-Ankle
                    - **Ankle**: Knee-Ankle-Foot
                    
                    **General Guidelines:**
                    - **~180°**: Fully extended/straight
                    - **~90°**: Right angle bend
                    - **<90°**: Deep flexion
                    - **>90°**: Partial extension
                    """)
                
                # Processing controls
                st.subheader("⚡ Processing")
                
                if st.button("🔄 Process Key Frames", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process every 20th frame for efficiency
                    step = max(1, frame_count // 50)
                    processed = 0
                    total_to_process = len(range(0, frame_count, step))
                    
                    for i in range(0, frame_count, step):
                        if i not in st.session_state.angle_data:
                            frame = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, i)
                            
                            if frame is not None:
                                annotated_frame, angles = st.session_state.analyzer.process_frame(frame)
                                
                                st.session_state.angle_data[i] = {
                                    'annotated_frame': annotated_frame,
                                    'angles': angles
                                }
                        
                        processed += 1
                        progress = processed / total_to_process
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {i}/{frame_count}")
                    
                    progress_bar.progress(1.0)
                    st.success("✅ Key frames processed!")
                
                # Export data
                if st.session_state.angle_data and st.button("📊 Export All Data", use_container_width=True):
                    # Create export data
                    export_text = "Frame,Time(s),L_Elbow,L_Hip,L_Knee,L_Ankle,R_Elbow,R_Hip,R_Knee,R_Ankle\n"
                    
                    for frame_num in sorted(st.session_state.angle_data.keys()):
                        angles = st.session_state.angle_data[frame_num]['angles']
                        time_s = frame_num / fps
                        
                        # Format angles (N/A if None)
                        angle_values = []
                        for side in ['left', 'right']:
                            for joint in ['elbow', 'hip', 'knee', 'ankle']:
                                angle = angles[f'{side}_{joint}']
                                angle_values.append(f"{angle:.1f}" if angle is not None else "N/A")
                        
                        export_text += f"{frame_num},{time_s:.3f},{','.join(angle_values)}\n"
                    
                    st.code(export_text, language="csv")
                    st.info("💡 Copy the data above and save as .csv file")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    
    else:
        # Instructions when no video loaded
        st.markdown("""
        ### 📋 How to Use:
        
        1. **📤 Upload Video**: Choose a video file with clear body movements
        2. **🎬 Navigate**: Use slider and controls to move through frames
        3. **📐 View Angles**: See all 8 joint angles in real-time
        4. **⚡ Process**: Batch process key frames for analysis
        5. **📊 Export**: Get complete angle data as CSV
        
        ### 🎯 Tracked Joints (Both Sides):
        - **🦾 Elbow**: Shoulder-Elbow-Wrist angle
        - **🦴 Hip**: Shoulder-Hip-Knee angle  
        - **🦵 Knee**: Hip-Knee-Ankle angle
        - **🦶 Ankle**: Knee-Ankle-Foot angle
        
        ### 💡 Best Results:
        - **Full body visible** in frame
        - **Good lighting** and contrast
        - **Clear joint landmarks** 
        - **Front or side view** works best
        
        ### 📊 Applications:
        - **Exercise form analysis**
        - **Gait analysis**
        - **Sports biomechanics**
        - **Physical therapy assessment**
        - **Movement rehabilitation**
        """)

if __name__ == "__main__":
    main()
