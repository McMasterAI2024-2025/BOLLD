import os
import sys
# Add lip_to_text directory to path for imports
lip_to_text_dir = os.path.join(os.path.dirname(__file__), 'lip_to_text')
sys.path.append(lip_to_text_dir)
import streamlit as st
import mediapipe as mp
import cv2
import pandas as pd
import pickle
import numpy as np
import warnings
import plotly.graph_objects as go
from collections import deque
import tensorflow as tf
import dlib
from lip_to_text.utils import num_to_char
from lip_to_text.modelutils import load_model as load_lip_model
warnings.filterwarnings("ignore")

SHAPE_PREDICTOR_PATH = os.path.join("lip_to_text", "shape_predictor_68_face_landmarks.dat")
BODY_MODEL_PATH = os.path.join('body_language_decoder', 'body_lang_model.pkl')


# Initialize session state variables
if 'threat_history' not in st.session_state:
    st.session_state.threat_history = deque(maxlen=100)
    st.session_state.q_table = {}
    st.session_state.action_history = deque(maxlen=100)
    st.session_state.reward_history = deque(maxlen=100)
    st.session_state.running = False
    st.session_state.lip_frames = []
    st.session_state.last_transcription = ""
    st.session_state.last_violence_value = 0

# Violence dictionary
dictionary = {
    "fuckyou": 1,
    "fuck": 0.8,
    "killyou": 1,
    "kill": 0.5,
    "hate": 0.5,
    "screwyou": 1,
    "fucking": 0.8,
    "stupid": 0.5,
    "foue": 0.2
}

# config the page
st.set_page_config(page_title="Threat Detection System", layout="wide")

# create sidebar
# st.sidebar.title("Settings")
# learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
# discount_factor = st.sidebar.slider("Discount Factor", 0.1, 0.99, 0.9)
# epsilon = st.sidebar.slider("Exploration Rate", 0.0, 1.0, 0.2)
learning_rate = 0.28
discount_factor = 0.71
epsilon = 0.39

# title
st.markdown("<h1 style='text-align: center; '>BOLLD</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Body and Oral Language Learning Decoder</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Real-time Threat Detection System</h2>", unsafe_allow_html=True)

# buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)
start_button = col1.button('Start', key='start_button')
stop_button = col6.button('Stop', key='stop_button')

# execute buttons
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# init MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# loading the pre-trained body language model
@st.cache_resource
def load_body_language_model():
    try:
        with open(BODY_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file '{BODY_MODEL_PATH}' not found.")
        return None

# Load and verify all required models/files
try:
    body_model = load_body_language_model()
    lip_model = load_lip_model()  # This will load from models/new_best_weights2.weights.h5
    
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        st.error(f"Shape predictor file not found at {SHAPE_PREDICTOR_PATH}")
        st.stop()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# threat probabilities buffer
threat_probs_window = [[0.5, 0.5] for _ in range(40)]
actions = ["all-good", "de-escalate"]

def rolling_threat_average(threat_probs):
    global threat_probs_window
    threat_probs_window.insert(0, threat_probs)
    threat_probs_window = threat_probs_window[:40]
    return np.mean(threat_probs_window, axis=0)

def normalize_and_scale_landmarks(frame_landmarks, ref_lm_ind, left_lm_ind, right_lm_ind):
    if frame_landmarks is None or frame_landmarks.landmark is None:
        return [0] * (33 * 4)  # return zeros for missing landmarks
    
    body_frame = frame_landmarks.landmark
    reference_landmark = body_frame[ref_lm_ind]
    ref_x, ref_y, ref_z = reference_landmark.x, reference_landmark.y, reference_landmark.z
    right_lm = body_frame[right_lm_ind]
    left_lm = body_frame[left_lm_ind]
    scale_factor = np.sqrt(
        (right_lm.x - left_lm.x) ** 2 +
        (right_lm.y - left_lm.y) ** 2 +
        (right_lm.z - left_lm.z) ** 2
    )
    if scale_factor == 0:
        scale_factor = 1e-6
    return list(
        np.array([
            [
                (landmark.x - ref_x) / scale_factor,
                (landmark.y - ref_y) / scale_factor,
                (landmark.z - ref_z) / scale_factor,
                landmark.visibility
            ]
            for landmark in body_frame
        ]).flatten()
    )

# Lip reading functions
def violence_classify(prediction):
    words = prediction.split(" ")
    violence_max = 0
    for word in words:
        if word in dictionary:
            if dictionary[word] > violence_max:
                violence_max = dictionary[word]
    return violence_max

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

# gets current state based on the numerical body lang value
def get_state(threatness_level):
    if threatness_level < 0.4:
        return "low"
    elif 0.4 <= threatness_level <= 0.7:
        return "medium"
    else:
        return "high"

# based on the state it selects the action that must be executed
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    if state in st.session_state.q_table:
        return max(st.session_state.q_table[state], key=st.session_state.q_table[state].get)
    return np.random.choice(actions)

# updates the q table based on all the current values
def update_q_table(state, action, reward, next_state):
    if state not in st.session_state.q_table:
        st.session_state.q_table[state] = {a: 0 for a in actions}
    if next_state not in st.session_state.q_table:
        st.session_state.q_table[next_state] = {a: 0 for a in actions}
    
    st.session_state.q_table[state][action] += learning_rate * (
        reward + discount_factor * max(st.session_state.q_table[next_state].values()) - 
        st.session_state.q_table[state][action]
    )

# placeholder for video feed
video_placeholder = st.empty()

# columns for metrics to be displayed
col1, col2, col3, col4 = st.columns(4)
threat_metric = col1.empty()
action_metric = col2.empty()
# transcription_metric = col3.empty()
violence_metric = col3.empty()
q_values_metric = col4.empty()

# placeholder for the graph 
graph_placeholder = st.empty()

# graph that shows the threat level and the reward
def update_graphs():
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            y=list(st.session_state.threat_history),
            name="Threat Level",
            line=dict(color="red")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            y=list(st.session_state.reward_history),
            name="Reward",
            line=dict(color="green")
        )
    )
    
    fig.update_layout(
        title="Threat Level and Reward History",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    graph_placeholder.plotly_chart(fig, use_container_width=True)

# Load models
body_model = load_body_language_model()
lip_model = load_lip_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lip_to_text/shape_predictor_68_face_landmarks.dat")

# main function that runs the threat detection system
if body_model is not None and lip_model is not None and st.session_state.running:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened() and st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error: Could not read frame from webcam.")
                        break
                    
                    # Process frame for both body language and lip reading
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    
                    # Get lip region and process for lip reading
                    lip_region = get_lip_region(image, detector, predictor)
                    if lip_region is not None:
                        lip_frame = preprocess_lip_frame(tf.image.rgb_to_grayscale(lip_region))
                        st.session_state.lip_frames.append(lip_frame)
                    
                    # Process lip frames when we have enough
                    if len(st.session_state.lip_frames) >= 75:
                        # print(st.session_state.lip_frames)
                        input_data = tf.stack(st.session_state.lip_frames)
                        mean = tf.math.reduce_mean(input_data)
                        std = tf.math.reduce_std(tf.cast(input_data, tf.float32))
                        input_data = tf.cast((input_data - mean), tf.float32) / std
                        
                        yhat = lip_model.predict(tf.expand_dims(input_data, axis=0))
                        decoder = tf.keras.backend.ctc_decode(yhat, [75], beam_width=75)[0][0].numpy()
                        prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")
                        
                        # Clean up prediction text
                        prediction = prediction.replace("foue", "fuck").replace("fiun", "fuck")
                        prediction = prediction.replace("slasin", "kill").replace("slain", "kill")
                        prediction = prediction.replace("seiu", "shoot")
                        prediction = prediction.replace("siux sobu", "screwyou")
                        prediction = prediction.replace("soue", "you").replace("sobu", "you")
                        
                        st.session_state.last_transcription = prediction
                        st.session_state.last_violence_value = violence_classify(prediction)
                        st.session_state.lip_frames = st.session_state.lip_frames[15:]
                    
                    # Process body language
                    try:
                        # Normalize landmarks and make prediction
                        normalized_pose_row = normalize_and_scale_landmarks(results.pose_landmarks, 0, 11, 12)
                        normalized_face_row = normalize_and_scale_landmarks(results.face_landmarks, 0, 33, 263)
                        normalized_right_hand_row = normalize_and_scale_landmarks(results.right_hand_landmarks, 0, 4, 20) if results.right_hand_landmarks else [0] * (21 * 4)
                        normalized_left_hand_row = normalize_and_scale_landmarks(results.left_hand_landmarks, 0, 4, 20) if results.left_hand_landmarks else [0] * (21 * 4)
                        row = normalized_pose_row + normalized_face_row + normalized_right_hand_row + normalized_left_hand_row
                        
                        X = pd.DataFrame([row])
                        body_language_prob = body_model.predict_proba(X)[0]
                        
                        # Combine body language and speech threats
                        combined_threat = (body_language_prob[1] + st.session_state.last_violence_value) / 2
                        state = get_state(combined_threat)
                        action = choose_action(state)
                        
                        # Calculate reward based on combined threat
                        reward = -1 if (action == "all-good" and combined_threat < 0.5) else 1
                        
                        # Update histories and Q-table
                        st.session_state.threat_history.append(combined_threat)
                        st.session_state.action_history.append(action)
                        st.session_state.reward_history.append(reward)
                        
                        next_state = get_state(combined_threat)
                        update_q_table(state, action, reward, next_state)
                        
                        # Update UI
                        threat_metric.metric("Threat Level", f"{combined_threat:.2f}")
                        action_metric.metric("Current Action", action)
                        # transcription_metric.metric("Transcription", st.session_state.last_transcription)
                        violence_metric.metric("Violence Value", f"{st.session_state.last_violence_value:.2f}")
                        q_values_metric.metric("Q-Values", str(st.session_state.q_table.get(state, {})))
                        
                        # Update graphs
                        update_graphs()
                        
                        # Add visual feedback
                        status_message = "Warning: Elevated Threat Level" if combined_threat > 0.5 else "Status: Normal"
                        color = (0, 0, 255) if combined_threat > 0.5 else (0, 255, 0)
                        cv2.putText(image, status_message, (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                        
                        # Display video
                        video_placeholder.image(image, channels="RGB", use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error processing frame: {str(e)}")
                        
            cap.release()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display Q-table
st.header("Q-Learning Table")
st.dataframe(pd.DataFrame.from_dict(st.session_state.q_table, orient='index'))