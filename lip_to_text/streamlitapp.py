import os
import tensorflow as tf
import imageio
import streamlit as st
from utils import num_to_char
from modelutils import load_model
import numpy as np
import cv2
import dlib
from typing import List

def get_lip_region(frame, detector, predictor):
    """Extract lip region using facial landmarks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    
    if not faces:
        return None
    
    # Get the first face
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # Extract lip landmarks (points 48-68 in dlib's 68 point model)
    lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 68)])
    
    # Get the bounding box of lips with padding
    x_min = np.min(lip_points[:, 0]) - 10
    x_max = np.max(lip_points[:, 0]) + 10
    y_min = np.min(lip_points[:, 1]) - 10
    y_max = np.max(lip_points[:, 1]) + 10
    
    # Ensure coordinates are within frame bounds
    x_min = max(0, x_min)
    x_max = min(frame.shape[1], x_max)
    y_min = max(0, y_min)
    y_max = min(frame.shape[0], y_max)
    
    # Extract and resize lip region to match expected input size (40x140)
    lip_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    if lip_region.size == 0:
        return None
    
    lip_region = cv2.resize(lip_region, (140, 46))
    return lip_region

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# Main Title
st.markdown("<h1 style='text-align: center;'>👄 LipBuddy: Your Lip Reading Companion</h1>", unsafe_allow_html=True)

# Setup the sidebar
with st.sidebar:
    st.markdown("## Welcome to LipBuddy!")
    st.markdown("""
**👋 Hello there!** 

1. **Start your camera**: The camera feed will be displayed in real-time.
2. **Preview the video**: See yourself on the left side.
3. **Model Visualization**: On the right, see what the ML model "sees" and the predicted tokens.
4. **Final Transcription**: Get a decoded transcript of the video.

**Pro Tips**:
- Make sure you're well-lit and facing the camera
- Keep your face centered and steady
- Speak clearly and at a moderate pace
    """)

col1, col2 = st.columns(2)

# Initialize face detection
detector = dlib.get_frontal_face_detector()
# You'll need to download the shape predictor file and specify its path
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the camera
cap = cv2.VideoCapture(0)
frames = []
count = 0

# Load the model
model = load_model()
if model is None:
    st.error("🚨 Failed to load the model. Please check your configuration.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Create placeholders
with col1:
    st.markdown("### 📺 Camera Feed")
    camera_placeholder = st.empty()

with col2:
    st.markdown("### 🔬 Model Input Visualization")
    visualization_placeholder = st.empty()
    prediction_placeholder = st.markdown("")
    status_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from camera")
        break

    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the camera feed with face rectangle and landmarks
    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # Get lip region
    lip_region = get_lip_region(frame_rgb, detector, predictor)
    
    if lip_region is not None:
        # Convert to grayscale and add to frames
        lip_frame = tf.image.rgb_to_grayscale(lip_region)
        frames.append(lip_frame)
        status_placeholder.success("Face detected - tracking lips")
    else:
        status_placeholder.warning("No face detected - please center your face in the camera")
        continue

    # Process when we have enough frames
    if len(frames) == 150:
        count += 1
        
        # Preprocess frames
        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        input_data = tf.cast((frames - mean), tf.float32) / std

        # Prepare visualization
        viz_frames = [np.squeeze(frame.numpy(), axis=-1) for frame in input_data]
        viz_frames = [np.uint8(frame * 255) for frame in viz_frames]
        
        # Save and display the visualization
        imageio.mimsave('output.gif', viz_frames, fps=10)
        visualization_placeholder.image('output.gif', use_container_width=True)

        # Run prediction
        yhat = model.predict(tf.expand_dims(input_data, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        
        # Convert prediction to text
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
        prediction_placeholder.markdown(f"### 🗣️ Predicted Text\n`{converted_prediction}`")

        # Remove oldest frame
        frames.pop(0)

        # Optional: Stop after certain number of predictions
        if count >= 5:
            break

# Clean up
cap.release()