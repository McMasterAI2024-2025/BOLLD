# 🧠 Body and
# 🗣️ Oral
# 📝 Language
# 📚 Learning
# 🧩 Decoder

## B.O.L.L.D leverages body language, lip transcriptions and reinforcement learning to detect threats in real-time.

### Team Members:
- Nicole Sorokin
- Julia Brzustowski
- Zuhair Qureshi
- Grady Rueffer
- Sophia Shantharupan

### Use Cases:
- Security applications: Detecting threats or violent language when audio is corrupted or unavailable during meetings.
- Violence mitigation: This project can potentially be used for public safety, such as in campus surveillance or implemented in glasses with cameras of something of that sort to aid people with disabilities such as blindness to be notified of any potential threats they may not be able to see.

### ...add tech stack here

run the app using the following command
```

streamlit run app.py
```

Currently the app.py contains the body language code *training and details about which can be found in the body_lang_decoder folder*

The lip to text component is under construction and will be added soon.

Additionally, the app.py contains reinforcement learning code, details below:

State Space:
```
def get_state(threatness_level):
    if threatness_level < 0.4:
        return "low"
    elif 0.4 <= threatness_level <= 0.7:
        return "medium"
    else:
        return "high"
```
State space is simplified into three levels (low, medium, high) based on the threat probability from the body language model. This simplifcation allows the learningto be more manageable while still capturing the essential threat levels.

Action Space:
```
actions = ["escalate", "de-escalate"]
```
Action space is simplified into two actions (escalate and de-escalate) based on the current state. This simplifies the learning process as well as the decision making process.

Q-Learning Table:
```
def update_q_table(state, action, reward, next_state):
    if state not in st.session_state.q_table:
        st.session_state.q_table[state] = {a: 0 for a in actions}
    if next_state not in st.session_state.q_table:
        st.session_state.q_table[next_state] = {a: 0 for a in actions}
    
    st.session_state.q_table[state][action] += learning_rate * (
        reward + discount_factor * max(st.session_state.q_table[next_state].values()) - 
        st.session_state.q_table[state][action]
    )
```
The Q-Learning table is a dictionary that stores the Q-values for each state-action pair. The Q-values are updated based on the current state, action, reward, and next state. The Q-values are used to determine the best action to take in the next state.

Action Selection:
```
def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    if state in st.session_state.q_table:
        return max(st.session_state.q_table[state], key=st.session_state.q_table[state].get)
    return np.random.choice(actions)
```
The action selection process is based on the current state and the Q-Learning table. The action selection process is random if the exploration rate is high, and based on the Q-values if the exploration rate is low. The Q-values are updated based on the current state, action, reward, and next state.

Reward Calculation:
```
if action == "escalate":
    reward = -1 if threatness_level < 0.5 else 1
else:
    reward = 1 if threatness_level < 0.5 else -1
```
The reward calculation is based on the current action and the threat probability. (Add more info on how the reward is calculated.)

### Benefits of using reinforcement learning:
- Learning from trial and error it improves the accuracy of the model.
- It allows for adaptation to new situations.
- The reward system provides immediate feedback about the appropriateness of actions.
- Continuously improve its decision-making based on experience.
- and many more to be added soon...


### Next steps:
1. Evaluate the performance of the reinforcement learning model and create some graphs to visualize the learning process.
2. Create a decision making tree that shows all th possible actions and their outcomes and how the rl model learns from this/chooses its actions.
3. Update research doc.
4. Update design doc.
5. Update process flow diagram.
