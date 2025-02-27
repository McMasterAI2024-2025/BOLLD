import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import pandas as pd

from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
import csv
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

with open('body_lang_model.pkl', 'rb') as f:
    model = pickle.load(f)


threat_probs_window = [[0.5,0.5] for _ in range(50)]

def rolling_threat_average(threat_probs):
    global threat_probs_window
    threat_probs_window.insert(0, threat_probs)
    threat_probs_window = threat_probs_window[:50]
    return np.mean(threat_probs_window, axis=0)

def normalize_and_scale_landmarks(frame_landmarks, ref_lm_ind, left_lm_ind, right_lm_ind):
    # Extract frame landmarks
    body_frame = frame_landmarks.landmark

    # Select a reference landmark
    reference_landmark = body_frame[ref_lm_ind]

    # Get the reference point's coordinates
    ref_x, ref_y, ref_z = reference_landmark.x, reference_landmark.y, reference_landmark.z


    # Calculate scale factor (e.g., distance between right and left landmarks for size normalization)
    right_lm = body_frame[right_lm_ind]
    left_lm = body_frame[left_lm_ind]

    scale_factor = np.sqrt(
        (right_lm.x - left_lm.x) ** 2 +
        (right_lm.y - left_lm.y) ** 2 +
        (right_lm.z - left_lm.z) ** 2
    )

    # Avoid division by zero
    if scale_factor == 0:
        scale_factor = 1e-6

    # Normalize landmarks
    normalized_body_frame_row = list(
        np.array([
            [
                (landmark.x - ref_x) / scale_factor,  # Normalize relative x
                (landmark.y - ref_y) / scale_factor,  # Normalize relative y
                (landmark.z - ref_z) / scale_factor,  # Normalize relative z
                landmark.visibility                  # Visibility remains unchanged
            ]
            for landmark in body_frame
        ]).flatten()
    )

    return normalized_body_frame_row

emotion_val = 0
i = 0
cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Export coordinates
            # try:
            # Extract Pose landmarks
            try:
                pose = results.pose_landmarks.landmark
                normalized_pose_row = normalize_and_scale_landmarks(results.pose_landmarks, 0, 11, 12)
                #pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                normalized_face_row = normalize_and_scale_landmarks(results.face_landmarks, 0, 33, 263)

                right_hand = results.right_hand_landmarks
                left_hand = results.left_hand_landmarks

                right_hand = results.right_hand_landmarks
                left_hand = results.left_hand_landmarks
                normalized_right_hand_row = [0 for _ in range(21*4)]
                normalized_left_hand_row = [0 for _ in range(21*4)]

                if right_hand:
                    normalized_right_hand_row = normalize_and_scale_landmarks(right_hand, 0, 4, 20)

                if left_hand:
                    normalized_left_hand_row = normalize_and_scale_landmarks(left_hand, 0, 4, 20)
                   
                # Concatenate rows
                row = normalized_pose_row+normalized_face_row+normalized_right_hand_row+normalized_left_hand_row
                # Make Detections
                try:
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]

                except Exception as e:
                    print(e)
                
                if rolling_threat_average(body_language_prob)[1] > 0.5:
                    body_language_class = 'Threatening'
                else:
                    body_language_class = 'Non-threatening'

                #print(body_language_class, body_language_prob, "HELLO")
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (500, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(np.round(rolling_threat_average(body_language_prob)[np.argmax(rolling_threat_average(body_language_prob))], 2))  # round(body_language_prob[np.argmax(body_language_prob)],2)
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if body_language_class.split(' ')[0] == 'Threatening':
                    emotion_val = round(-1*body_language_prob[np.argmax(body_language_prob)],2)

                elif body_language_class.split(' ')[0] == 'Non-threatening':
                    emotion_val = round(body_language_prob[np.argmax(body_language_prob)],2)

                print("Probs:",body_language_prob)
            
            except:
                pass
                            
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break

cap.release()
cv2.destroyAllWindows()

