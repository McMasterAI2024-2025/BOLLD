# Body Language Detection with Mediapipe and OpenCV

This code captures live webcam footage and uses a pre-trained machine learning model to classify body language based on facial and pose landmarks.

## Key Components
- **Libraries Used**: 
  - `mediapipe` for detecting and drawing landmarks on the face and body.
  - `OpenCV` for handling webcam input and displaying the video feed.
  - `pandas`, `pickle`, and `sklearn` for loading the pre-trained model and processing the predictions.

## How It Works
1. **Load Pre-trained Model**: Loads a pre-trained body language classification model from `body_lang_model.pkl`.
2. **Capture Webcam Input**: Uses OpenCV to capture video frames from the webcam.
3. **Detect Landmarks**: 
   - Mediapipe's Holistic model detects facial and pose landmarks.
   - Coordinates of landmarks are extracted, converted to numerical values, and passed to the pre-trained model for classification.
4. **Classification & Display**:
   - The model classifies the detected body language and calculates prediction probabilities.
   - Overlayed text and rectangles on the webcam feed display the predicted class and confidence level.

## Controls
- Press **'q'** to exit the webcam feed.

---

This setup is designed to detect and classify body language patterns in real-time.
