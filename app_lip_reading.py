# THIS FILE CONTAINS THE BEGINNING STAGES OF IMPLEMENTING LIP READING USING PHOMENE PATTERNS

# imports
import streamlit as st
import mediapipe as mp
import cv2
import pandas as pd
import pickle
import numpy as np
import warnings
import plotly.graph_objects as go
from collections import deque
warnings.filterwarnings("ignore")
from collections import deque

# dict of threatening words and their threat scores (0-1)
THREAT_WORDS = {
    # curse words medium threat
    "fuck": 0.6,
    "shit": 0.4,
    "damn": 0.3,
    
    # violent words high threat
    "kill": 0.9,
    "fight": 0.7,
    "punch": 0.8,
    "hit": 0.7,
    
    # threatening phrases high threat
    "going to hurt": 0.9,
    "make you pay": 0.8,
    "watch out": 0.7,
    
    # some aggressive words medium threat
    "hate": 0.6,
    "destroy": 0.7,
    "break": 0.5
}

# lip patterns for phonemes (simplified representation) need to make this better
PHONEME_PATTERNS = {
    # just some letters
    'f': {'upper_lip_height': 'small', 'inner_lip_gap': 'large', 'mouth_width': 'small', 'lip_curvature': 'flat', 'mouth_asymmetry': 'symmetric'},
    'u': {'upper_lip_height': 'medium', 'inner_lip_gap': 'large', 'mouth_width': 'small', 'lip_curvature': 'frown', 'mouth_asymmetry': 'asymmetric'}
    # 'k': {'upper_lip_height': 'small', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'h': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 't': {'upper_lip_height': 'small', 'lower_lip_height': 'small', 'mouth_width': 'medium'},
    
    # # vowels
    # 'a': {'upper_lip_height': 'large', 'lower_lip_height': 'large', 'mouth_width': 'wide'},
    # 'e': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'wide'},
    # 'i': {'upper_lip_height': 'medium', 'lower_lip_height': 'small', 'mouth_width': 'medium'},
    # 'o': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'small'},
    # 'u': {'upper_lip_height': 'small', 'lower_lip_height': 'small', 'mouth_width': 'small'},

    # # more letters and sound combinations
    # 'b': {'upper_lip_height': 'small', 'lower_lip_height': 'small', 'mouth_width': 'medium'},
    # 'p': {'upper_lip_height': 'small', 'lower_lip_height': 'small', 'mouth_width': 'medium'},
    # 'm': {'upper_lip_height': 'small', 'lower_lip_height': 'small', 'mouth_width': 'medium'},
    # 's': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'z': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'th': {'upper_lip_height': 'small', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'v': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'r': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'wide'},
    # 'w': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'small'},
    # 'l': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'sh': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'ch': {'upper_lip_height': 'medium', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
    # 'g': {'upper_lip_height': 'small', 'lower_lip_height': 'medium', 'mouth_width': 'medium'},
}

# init session state variables
if 'threat_history' not in st.session_state:
    st.session_state.threat_history = deque(maxlen=100)
    st.session_state.q_table = {}
    st.session_state.action_history = deque(maxlen=100)
    st.session_state.reward_history = deque(maxlen=100)
    st.session_state.running = False
    st.session_state.word_buffer = deque(maxlen=10)  # Buffer for detected phonemes
    st.session_state.last_oral_threat = 0.0  # Last detected oral threat level


# config the page
st.set_page_config(page_title="Threat Detection System", layout="wide")

# create sidebar
st.sidebar.title("Settings")
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
discount_factor = st.sidebar.slider("Discount Factor", 0.1, 0.99, 0.9)
epsilon = st.sidebar.slider("Exploration Rate", 0.0, 1.0, 0.2)

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
def load_model():
    try:
        with open('machine_learning/body_language_decoder/body_lang_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'body_lang_model.pkl' not found. Please ensure the model file is in the correct location.")
        return None

model = load_model()

# threat probabilities buffer
threat_probs_window = [[0.5, 0.5] for _ in range(40)]
actions = ["escalate", "de-escalate"]

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

#functions for threat detection using phonemes
def analyze_lip_movement(face_landmarks):
    """Analyze lip movements to detect potential phonemes"""
    if not face_landmarks or not face_landmarks.landmark:
        return None
    
    # Key lip landmarks indices
    upper_lip_center = face_landmarks.landmark[13]  # Upper lip center
    lower_lip_center = face_landmarks.landmark[14]  # Lower lip center
    inner_upper_lip_center = face_landmarks.landmark[62]  # Inner upper lip center
    inner_lower_lip_center = face_landmarks.landmark[66]  # Inner lower lip center
    outer_left_corner = face_landmarks.landmark[78]  # Outer left corner
    outer_right_corner = face_landmarks.landmark[308]  # Outer right corner
    inner_left_corner = face_landmarks.landmark[61]  # Inner left corner
    inner_right_corner = face_landmarks.landmark[291]  # Inner right corner
    mid_upper_curve = face_landmarks.landmark[50]  # Midpoint of upper lip curve
    mid_lower_curve = face_landmarks.landmark[280]  # Midpoint of lower lip curve

    # print("Upper lip center:", upper_lip_center)
    # print("Lower lip center:", lower_lip_center)
    # print("Inner upper lip center:", inner_upper_lip_center)
    # print("Inner lower lip center:", inner_lower_lip_center)
    # print("Outer left corner:", outer_left_corner)
    # print("Outer right corner:", outer_right_corner)
    # print("Inner left corner:", inner_left_corner)
    # print("Inner right corner:", inner_right_corner)
    # print("Mid upper curve:", mid_upper_curve)
    # print("Mid lower curve:", mid_lower_curve)

    # Calculate mouth measurements
    mouth_height = abs(upper_lip_center.y - lower_lip_center.y)
    mouth_width = abs(outer_right_corner.x - outer_left_corner.x)
    lip_curvature = abs(mid_upper_curve.y - mid_lower_curve.y)
    inner_lip_gap = abs(inner_upper_lip_center.y - inner_lower_lip_center.y)  # Gap between inner lips
    outer_lip_gap = abs(upper_lip_center.y - lower_lip_center.y)  # Gap for outer lips
    lip_curvature = abs(mid_upper_curve.y - mid_lower_curve.y)  # Curvature of lips
    mouth_asymmetry = abs(inner_left_corner.y - inner_right_corner.y)  # Asymmetry of lip corners


    # print("Mouth height:", mouth_height)
    # print("Mouth width:", mouth_width)
    
    # Classify mouth shape
    shape = {
        'upper_lip_height': 'medium' if 0.02 < outer_lip_gap < 0.04 else 'small' if outer_lip_gap <= 0.02 else 'large',
        'inner_lip_gap': 'medium' if 0.01 < inner_lip_gap < 0.03 else 'small' if inner_lip_gap <= 0.01 else 'large',
        'mouth_width': 'medium' if 0.3 < mouth_width < 0.5 else 'small' if mouth_width <= 0.3 else 'wide',
        'lip_curvature': 'flat' if lip_curvature < 0.005 else 'smile' if mid_upper_curve.y < mid_lower_curve.y else 'frown',
        'mouth_asymmetry': 'symmetric' if mouth_asymmetry < 0.005 else 'asymmetric',
    }

    print("Detected Shape:", shape)
    
    # Match shape to phonemes
    matched_phonemes = []
    for phoneme, pattern in PHONEME_PATTERNS.items():
        if all(shape[k] == v for k, v in pattern.items()):
            matched_phonemes.append(phoneme)
    
    return matched_phonemes[0] if matched_phonemes else None

def detect_threatening_words(phoneme):
    """Detect threatening words from phoneme sequence"""
    if 'word_buffer' not in st.session_state:
        st.session_state.word_buffer = deque(maxlen=10)  # Safety check

    if phoneme:
        st.session_state.word_buffer.append(phoneme)
    
    # Convert buffer to string
    current_sequence = ''.join(list(st.session_state.word_buffer))
    # print("Current sequence:", current_sequence)
    
    # Check for threatening words
    max_threat = 0.0
    for word, threat_level in THREAT_WORDS.items():
        if word in current_sequence.lower():
            max_threat = max(max_threat, threat_level)
            st.session_state.word_buffer.clear()  # Reset buffer after detection
    
    return max_threat

def combine_threat_levels(body_threat, oral_threat):
    """Combine body language and oral threat levels"""
    # Weight factors (can be adjusted)
    body_weight = 0.6
    oral_weight = 0.4
    
    return (body_weight * body_threat) + (oral_weight * oral_threat)

# placeholder for video feed
video_placeholder = st.empty()

# columns for metrics to be displayed
col1, col2, col3 = st.columns(3)
threat_metric = col1.empty()
action_metric = col2.empty()
q_values_metric = col3.empty()

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

# main function that runs the threat detection system
if model is not None and st.session_state.running:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam. Please check your camera connection.")
        else:
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened() and st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Error: Could not read frame from webcam.")
                        break
                        
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        # process landmarks with error handling
                        normalized_pose_row = normalize_and_scale_landmarks(results.pose_landmarks, 0, 11, 12) if results.pose_landmarks else [0] * 132
                        normalized_face_row = normalize_and_scale_landmarks(results.face_landmarks, 0, 33, 263) if results.face_landmarks else [0] * 1872
                        normalized_right_hand_row = normalize_and_scale_landmarks(results.right_hand_landmarks, 0, 4, 20) if results.right_hand_landmarks else [0] * 84
                        normalized_left_hand_row = normalize_and_scale_landmarks(results.left_hand_landmarks, 0, 4, 20) if results.left_hand_landmarks else [0] * 84

                        row = normalized_pose_row + normalized_face_row + normalized_right_hand_row + normalized_left_hand_row
                        

                        # make the prediction
                        X = pd.DataFrame([row])
                        body_language_prob = model.predict_proba(X)[0]
                        body_threat_level = rolling_threat_average(body_language_prob)[1]
                        
                        # Process oral threats
                        detected_phoneme = analyze_lip_movement(results.face_landmarks)
                        print("Detected phoneme:", detected_phoneme)
                        oral_threat_level = detect_threatening_words(detected_phoneme)
                        st.session_state.last_oral_threat = max(st.session_state.last_oral_threat, oral_threat_level)
                        
                        # Combine threat levels
                        # combined_threat_level = combine_threat_levels(body_threat_level, st.session_state.last_oral_threat)
                        combined_threat_level = body_threat_level # for now dont add in lip movement threat level
                        
                        # updating the state and choosing the action
                        state = get_state(combined_threat_level)
                        action = choose_action(state)

                        # Calculate reward based on combined threat
                        if action == "escalate":
                            reward = -1 if combined_threat_level < 0.5 else 1
                        else:
                            reward = 1 if combined_threat_level < 0.5 else -1

                        # updating the histories
                        st.session_state.threat_history.append(combined_threat_level)
                        st.session_state.action_history.append(action)
                        st.session_state.reward_history.append(reward)


                        # update Q-table
                        next_state = get_state(rolling_threat_average(body_language_prob)[1])
                        update_q_table(state, action, reward, next_state)

                        # updating all the values
                        threat_metric.metric("Combined Threat Level", f"{combined_threat_level:.2f}")
                        action_metric.metric("Current Action", action)
                        q_values_metric.metric("Q-Values", str(st.session_state.q_table.get(state, {})))

                        # updating the graphs
                        update_graphs()

                        # add visual feedback in text form to the actual video screen
                        if combined_threat_level > 0.5:
                            status_message = "Warning: Elevated Threat Level"
                            color = (0, 0, 255)
                        else:
                            status_message = "Status: Normal"
                            color = (0, 255, 0)

                        cv2.putText(image, status_message, (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                        
                        # addding the oral threat level
                        if st.session_state.last_oral_threat > 0:
                            oral_message = f"Oral Threat Level: {st.session_state.last_oral_threat:.2f}"
                            cv2.putText(image, oral_message, (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


                        # displaying the video screen
                        video_placeholder.image(image, channels="BGR", use_container_width=True)

                        # Decay oral threat level over time
                        st.session_state.last_oral_threat *= 0.95  # Gradual decay


                    except Exception as e:
                        st.error(f"Error processing frame: {str(e)}")

            cap.release()
    except Exception as e:
        st.error(f"Error: {str(e)}")

# displaying the Q-table
st.header("Q-Learning Table")
st.dataframe(pd.DataFrame.from_dict(st.session_state.q_table, orient='index'))