# Design Document

## Idea:  
**Lip Reading AI using Reinforcement Learning**  

## Use Cases
- Security applications: Detecting threats or violent language when audio is corrupted or unavailable during meetings.
- Violence mitigation: This project can potentially be used for public safety, such as in campus surveillance or implemented in glasses with cameras of something of that sort to aid people with disabilities such as blindness to be notified of any potential threats they may not be able to see.

## Tech stack: 
- Programming Language: Python 
- Datasets: numpy (data manipulations), subset of GRID, Shape Predictor 68
- Computer Vision: OpenCV (capturing video), Dlib (detecting lips), maybe media pipe also for detecting face, HandDetector
- Machine Learning Framework: Tensorflow
- Reinforcement Learning: Gymnasium (environment) 
- Frontend: React next.js 
- Backend: fastapi 
- Algorithms: Q-Learning

## The 2 Main Stages:  

### First:
- Employ conventional computer vision techniques classify a person's physical actions (both face and body language) and convert to a numerical value.
- Utilize distinct models to analyze lip movements to text and convert to a numerical value. 

### Second:  
- Based on the two inputs from the first stage, train a reinforcement learning model to recognize sequences of actions and lip movements that indicate malicious behavior (0 -> non-malicious, 1 -> malicious, and scale of 0-1 of 0 being non-threatening key-words and 1 being threatening key-words).
- Based on the agent, the state of the environment can change. De-escalate if correctly identified, escalate if incorrectly identified.

## Decisions + Documentation:
#### Body Language Detection:
- Using [EMOLIPS model]([url](https://github.com/SMIL-SPCRAS/EMOLIPS)) (CNN-LSTM model) to detect emotion from lips using details from the face.
- Negative emotions (e.g. anger, disgust) can be used to assist in threat identification.
- Oct-27: Changing to facial emotion recognition model using deepface.
- Integrating body language into threatening vs non-threatening classification [using mediapipe]([url](https://www.youtube.com/watch?v=We1uB79Ci-w)) -- train ML model on coordinates of landmarks in frames with associated labels.  
- Jan 13: Made the decision to use one body language model (mediapipe) due to multiprocessing conflicts of running two models at once (which was the original goal to get an average of both models).

#### Lip Movement to text:
- We will be closely following the methods of [LipNet](https://arxiv.org/pdf/1611.01599) as it has been proven to work and there is lots of existing documentations on this method
- This method uses Dlib for detecting facial landmarks and preprocessing the GRID dataset, then inputs a sequence of frames to 3 layers of 
of a CNN, each followed by a spatial max-pooling layer, then features are processed by two bidrectional GRUs; each time-step of the GRU output is processed by a linear layer and a softmax over the vocabulary. The model is then trained using CTC.
- Jan 13: Switching over to another model due to the previous one lacking the ability to process a live video stream. From here on out is to base our model off of something existing to transcribe lip movement to text (maybe whisper?) then use our own model to sleect words and compare against a dictionary or something to determine level of violence.

##### Preprocessing:
- Loaded video frames and detected faces and facial landmarks using dlib pretrained models
- Implemented mouth cropping based on mouth landmarks with reference point stabilization
- Uses a weighted average of the current and last frame
- Example (may need to open in a media player, not directly in code editor):
> [Uncropped Video](machine_learning/lip_reading/preprocessing/example/ex_1_uncropped.mp4)  
> [Cropped Video](machine_learning/lip_reading/preprocessing/example/ex_1_cropped.mp4)

## Rough Milestone Timelines:  
### Weeks 1-2:  
- Project kickoff and setup  
- Assign tasks  
- Define goals  
- Start preparing and exploring data, and implement processes to load it in  
- Create basic frontend and backend  
- Set up OpenCV  

### Weeks 3-4:  
- Split into two stages: lip reading and reinforcement learning
- Research different models/methods for both stages
- Start implementation

### Weeks 5-6:  
- Finish body language part of stage 1
- Set up RL environment
- Finish preprocessing for lip to text part of stage 1
- Continue implementation of training for lip to text part of stage 1

### Weeks 7-8:  
- Finish training for lip to text part of stage 1
- Finish RL stage 2
- Have a demo video done

### Weeks 8-10:  
- Connect stage 1 and 2 together
- Continue training the reinforcement learning model

## AFTER WINTER BREAK:  

### Weeks 11-13:  
- Building the frontend and backend and connecting with machine learning scripts
- Finalized body language model
- Finalize lip to text model
- Working on RL

### Weeks 13-14:  
- Working on connecting everything together

### Week 15:  
- Final touches  
- Maybe try to make it work with a webcam

## Process FLow Diagram:
# will be updated soon!
![Process Flow Diagram](/process_flow_dgm.jpg)