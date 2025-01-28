import os
import tensorflow as tf
import imageio
import streamlit as st
from utils import num_to_char
from modelutils import load_model
import numpy as np
import cv2
import dlib
import time
from typing import List
from collections import deque

def preprocess_lip_frame(frame):
    frame = tf.cast(frame, tf.float32)
    frame_np = frame.numpy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    frame_np = clahe.apply(frame_np.astype(np.uint8))
    frame_np = frame_np / 255.0
    return tf.convert_to_tensor(frame_np)

def get_lip_region(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if not faces:
        return None

    face = faces[0]
    landmarks = predictor(gray, face)
    outer_lips = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 60)])
    center_x = np.mean(outer_lips[:, 0])
    center_y = np.mean(outer_lips[:, 1])
    face_width = face.width()
    padding_x = int(face_width * 0.15)
    padding_y = int(face_width * 0.1)
    x_min = int(max(0, center_x - padding_x))
    x_max = int(min(frame.shape[1], center_x + padding_x))
    y_min = int(max(0, center_y - padding_y))
    y_max = int(min(frame.shape[0], center_y + padding_y))

    lip_region = frame[y_min:y_max, x_min:x_max]
    if lip_region.size == 0:
        return None
    target_size = (140, 46)
    lip_region = cv2.resize(lip_region, target_size)
    return lip_region

class PredictionSmoother:
    def __init__(self, window_size=3):
        self.predictions = deque(maxlen=window_size)
    
    def update(self, new_prediction):
        self.predictions.append(new_prediction)
    
    def get_smoothed_prediction(self):
        if not self.predictions:
            return None
        prediction_counts = {}
        for pred in self.predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        return max(prediction_counts.items(), key=lambda x: x[1])[0]

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>👄 Continuous Lip-to-Text</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

start_button = st.button("Start")
stop_button = st.button("Stop")

if "running" not in st.session_state:
    st.session_state.running = False

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

if st.session_state.running:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    smoother = PredictionSmoother()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frames = []
    model = load_model()
    if model is None:
        st.error("🚨 Failed to load the model. Please check your configuration.")
        st.stop()

    st.success("✅ Model loaded successfully!")

    with col1:
        st.markdown("### 📺 Camera Feed")
        camera_placeholder = st.empty()

    with col2:
        st.markdown("### 🔬 Model Input Visualization")
        visualization_placeholder = st.empty()
        prediction_placeholder = st.markdown("")
        status_placeholder = st.empty()

    confidence_placeholder = st.empty()
    fps_placeholder = st.empty()
    
    prev_time = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_placeholder.text(f"FPS: {fps:.2f}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        lip_region = get_lip_region(frame_rgb, detector, predictor)

        if lip_region is not None:
            lip_frame = preprocess_lip_frame(tf.image.rgb_to_grayscale(lip_region))
            frames.append(lip_frame)
            status_placeholder.success("Face detected - tracking lips")
        else:
            status_placeholder.warning("No face detected - please center your face in the camera")
            continue

        if len(frames) == 100:
            mean = tf.math.reduce_mean(frames)
            std = tf.math.reduce_std(tf.cast(frames, tf.float32))
            input_data = tf.cast((frames - mean), tf.float32) / std

            viz_frames = [
                ((frame.numpy() - frame.numpy().min()) * 255 / (frame.numpy().max() - frame.numpy().min())).astype(np.uint8)
                for frame in input_data
            ]
            imageio.mimsave('output.gif', viz_frames, fps=10)
            visualization_placeholder.image('output.gif', use_container_width=True)

            yhat = model.predict(tf.expand_dims(input_data, axis=0))
            confidence = np.max(yhat)
            confidence_placeholder.progress(float(confidence))

            decoder = tf.keras.backend.ctc_decode(yhat, [100], beam_width=100)[0][0].numpy()
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
            smoother.update(converted_prediction)
            smoothed_prediction = smoother.get_smoothed_prediction()

            if smoothed_prediction:
                prediction_placeholder.markdown(f"### 🗣️ Predicted Text\n`{smoothed_prediction}`")

            frames.clear()

    cap.release()