# 🧠 Body and
# 🗣️ Oral
# 📝 Language
# 📚 Learning
# 🧩 Decoder

## BOLLD employs a multi-modal approach, integrating body language analysis, lip transcriptions, and reinforcement learning to detect threats in real-time using computer vision and natural language processing.

### Team Members:
- [Nicole Sorokin](https://github.com/NicoleSorokin) (Team Lead)
- [Julia Brzustowski](https://github.com/juliaabrz) (Team Member)
- [Zuhair Qureshi](https://github.com/ZuhairQureshi) (Team Member)
- [Grady Rueffer](https://github.com/GradyRuefferOutlook) (Team Member)
- [Sophia Shantharupan](https://github.com/SophShan) (Team Member)

## 🚀 Use Cases  

### 🔒 **Security Applications**  
- Detecting potential **threats** or **violent language** when **audio** is corrupted or unavailable during meetings. 📞⚠️  
- Aimed at enhancing **safety** and providing an alternative threat detection system that doesn't rely on sound. 🎥🔍  

### 🛡️ **Violence Mitigation**  
- Can be applied in **public safety** scenarios, such as **campus surveillance**, to alert authorities of potential threats in real-time. 🎓🚨  
- **Assistive technology**: Can be implemented in **glasses with cameras** to help people with disabilities, like **blindness**, by notifying them of potential threats they might not visually perceive. 👓🤖👀  

## 🚀 Tech Stack  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Shape Predictor 68](https://img.shields.io/badge/Shape%20Predictor%2068-FF9800?style=for-the-badge)

### 🎥 Computer Vision
![OpenCV](https://img.shields.io/badge/OpenCV-E91E63?style=for-the-badge&logo=opencv&logoColor=white)
![Dlib](https://img.shields.io/badge/Dlib-9C27B0?style=for-the-badge)
![Mediapipe](https://img.shields.io/badge/Mediapipe-4CAF50?style=for-the-badge)

### 🤖 Machine Learning
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-2196F3?style=for-the-badge&logo=scikitlearn&logoColor=white)

### 📊 Visualization & Data Processing
![Matplotlib](https://img.shields.io/badge/Matplotlib-7B1FA2?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-03A9F4?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![CSV](https://img.shields.io/badge/CSV-757575?style=for-the-badge)

### 🧠 Algorithms
![Q-Learning](https://img.shields.io/badge/Q--Learning-FFEB3B?style=for-the-badge)

 

Run the app using the following command
```
streamlit run app.py
```

Currently the app.py contains the body language code *training and details about which can be found in the body_lang_decoder folder*, the lip transcription component *details about the model can be found in the lip_to_text folder* where each key word is compared to a list of threatening words and the threat level is calculated. The threat level is then used to determine the state of the system. Passed into the Q-Learning table, the state is used to determine the action to take using reinforcement learning.

## 🚀 High-Level Overview  

![process_flow_diagram](/assets/process_flow_diagram.drawio.png)

### 1️⃣ **First Stage**  
- Use a **trained body language model** 🕺 and **lip reading** (via Mediapipe landmarks) 👄 to compute a **numerical threat probability** (0-1) for each.  
- Combine both values to get a **combined threat score** 🔢.  

### 2️⃣ **Second Stage**  
- Based on the two inputs from the first stage, **train a reinforcement learning model** 🤖 to recognize sequences of actions and lip movements that suggest **malicious behavior**.  
  - **Output:** 0 ➔ **Non-malicious**, 1 ➔ **Malicious**, and a scale (0-1) representing the **threat level** of key words (0 = non-threatening, 1 = threatening).  
- The model will influence the **environment state** 🌍:  
  - **De-escalate** if the threat is correctly identified 🕊️.  
  - **All clear!** if the threat is incorrectly identified 🚨.  

## 📚 Decisions + Documentation  

### 🧠 **Body Language Detection**  
- Using the [**EMOLIPS model**](https://github.com/SMIL-SPCRAS/EMOLIPS) (CNN-LSTM) to detect emotions from lip movement based on face details. 👄😠  
- **Negative emotions** (e.g. anger, disgust) 🥴 can assist in identifying potential threats. ⚠️  
- **Oct 27**: Shifted to a **facial emotion recognition model** using **DeepFace** due to better performance. 🧑‍🎨  
- Integrating body language into a **threat vs. non-threat classification** using [**Mediapipe**](https://www.youtube.com/watch?v=We1uB79Ci-w). 🧑‍💻 The model trains on coordinates from landmarks in frames with associated labels.  
- **Jan 13**: Decided to use **one body language model** (Mediapipe) after facing **multiprocessing conflicts** with running two models simultaneously (initial goal was to get an average). 🤖❌  

### 👄 **Lip Movement to Text**  
- Closely following the methods of [**LipNet**](https://arxiv.org/pdf/1611.01599), as it's proven and well-documented. 📄  
- **Methodology**: Uses **Dlib** for facial landmark detection, preprocessing the GRID dataset, followed by a CNN architecture with bidirectional GRUs. CTC training used for model optimization. 📊  
- **Jan 13**: Switching models as the previous one couldn’t handle live video streams. Transitioning to a more suitable approach (e.g., **Whisper** model) to transcribe lip movement to text, then applying custom models to detect violence levels. 💻  
- **Jan 21**: Exploring a new technique using lip/mouth landmarks to detect **phonemes** and then identify key words stored in a dictionary with associated threat levels. 📖  
- **Jan 27**: Enhanced **LipNet** model to process **live video streams** 🎥 and detect mouth region with **Dlib + ShapePredictor68**.  
- **Jan 29**: Added algorithm to **detect key words** and produce a **violence value**. 🔑  
- **Jan 31**: Integrated into **app.py**. 🎉

## 📅 Rough Milestone Timelines  

### Weeks 1-2:  
- 🚀 **Project Kickoff**: Setup environment and tools  
- 👥 **Task Assignment**  
- 🎯 Define goals and objectives  
- 📊 Data exploration and preparation  
- 🌐 Create basic **frontend** & **backend**  
- 🎥 Set up **OpenCV** for video processing  

### Weeks 3-4:  
- 🔄 Split into **lip reading** and **reinforcement learning (RL)** stages  
- 🤔 Research different models and methods for both stages  
- 💻 Start implementation  

### Weeks 5-6:  
- ✅ Finish **body language** part of stage 1  
- 🌱 Set up **RL environment**  
- 📝 Finish preprocessing for **lip to text** part of stage 1  
- 🔄 Continue implementation of **lip to text** training  

### Weeks 7-8:  
- 🎓 Finish training **lip to text** part of stage 1  
- 🏁 Complete **RL** stage 2  
- 🎥 Create a **demo video**  

### Weeks 8-10:  
- 🔗 Connect stage 1 and 2  
- 🧠 Continue **reinforcement learning** model training  

### Weeks 11-13:  
- 🌐 **Frontend & Backend** integration with ML scripts  
- ✅ Finalize **body language model**  
- ✅ Finalize **lip to text model**  
- 🧠 Continue working on **RL**  

### Weeks 13-14:  
- 🔧 Finish **lip to text** model  
- 🔌 Integrate **lip to text** into the main **app.py**  

### Week 15:  
- ✨ Final touches  
- ⚙️ **Improve accuracy** and **fine-tuning**  
- 🖥️ Test the model with **webcam** integration  


### For more details, please refer to the [research document](BOLLD_Research_Document.pdf).