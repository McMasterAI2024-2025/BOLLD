import cv2
import os
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
import subprocess
import numpy as np
import pickle
import io
from starlette.background import BackgroundTask
import time

app = FastAPI()

# Path to your machine learning models
machine_learning_folder = os.path.join(os.getcwd(), "../machine_learning")
model_path = os.path.join(machine_learning_folder, 'body_language_decoder', 'body_lang_model.pkl')

# Load your model here to prevent reloading every time
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Start webcam
cap = cv2.VideoCapture(0)
frame_rate = 30

# Function to get threat detection
def get_threat_level(frame):
    # Process frame and extract the body language & lip-transcribed words
    # Perform your detection logic here, returning a numerical or classification output
    threat_value = 0.0  # Placeholder for the real output
    return threat_value

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe, etc.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get your threat level from the model
        threat_value = get_threat_level(frame_rgb)
        
        # Send the result as part of your data payload
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/compute-threat")
def compute_threat():
    # Placeholder for threat model - this will be dynamically fetched in real-time
    threat_message = get_threat_level('some_frame')  
    return {"message": f"Threat level: {threat_message}"}
