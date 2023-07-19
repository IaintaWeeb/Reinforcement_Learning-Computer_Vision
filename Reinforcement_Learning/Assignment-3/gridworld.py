import numpy as np
import matplotlib.pyplot as plt

wind = np.array([0,0,0,-1,-1,-1,-2,-2,-1,0])
moves = np.array([[1,0],[0,1],[-1,0],[0,-1]])
kingmoves = np.array([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])
n_col = 10
n_row = 7
n_state = n_col*n_row
start = [0,3]
end = [7,3]
n_episode = 100
epsilon = 0.1
alpha = 0.2
gamma = 1


def state_value_plot(Q,label):
    V = np.zeros(n_state)
    for i,state in enumerate(states):
        V[i]=np.max(Q[int(i)][:])
    V = np.reshape(V, (7,10))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V, cmap='cool')
    for (j,i),qval in np.ndenumerate(V):
        ax.text(i, j, np.round(qval,1), ha='center', va='center', fontsize=9)        
    plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.title(label)
    plt.savefig(label+'.png') 

def grid(moves,wind):
    Q = np.zeros((n_state,len(moves)))
    states=[]
    for i in range(n_row):
        for j in range(n_col):
            states.append(([j,i]))
    reward=np.zeros((n_state,len(moves)))
    next_states=np.zeros((n_state,len(moves)))
    for i,state in enumerate(states):
        for j,action in enumerate(moves):
            next_state = [max(min(state[0]+action[0],9),0),max(min(state[1]+action[1]+wind[state[0]],6),0)]
            if next_state in states:
                if next_state == end:
                    reward[i][j] = 100
                    next_states[i][j] = states.index(next_state)
                else: 
                    reward[i][j] = -1
                    next_states[i][j] = states.index(next_state)
            else:
                reward[i][j] = -100
                next_states[i][j] = states.index(state)
                Q[i][j] = -100
                
    return reward,next_states,states,Q

def action_sel(Q,state,moves):
    if np.random.random() > epsilon:
        return np.argmax(Q[int(state)][:])
    else:
        return np.random.choice(range(len(moves)))

def Sarsa(reward,next_states,states,Q,moves):
    t=[]
    t_=0
    for episode in range(n_episode):
        state = states.index(start)
        action = action_sel(Q,state,moves)
        while state != states.index(end):
            reward_ = reward[int(state)][int(action)]
            next_state_ = next_states[int(state)][int(action)]
            next_action = action_sel(Q,next_state_,moves)
            Q[int(state)][int(action)] = Q[int(state)][int(action)] + alpha*(reward_ + gamma*Q[int(next_state_)][int(next_action)] - Q[int(state)][int(action)])
            state = next_state_
            action = next_action
            t_ = t_ + 1
        t.append(t_)

    return t

def Exp_Sarsa(reward,next_states,states,Q,moves):
    t=[]
    t_=0
    for episode in range(n_episode):
        state = states.index(start)
        while state != states.index(end):
            action = action_sel(Q,state,moves)
            reward_ = reward[int(state)][int(action)]
            next_state_ = next_states[int(state)][int(action)]
            policy=np.ones(len(moves))*epsilon/len(moves)
            policy[np.argmax(Q[int(next_state_)][:])] = 1 - epsilon + epsilon/len(moves)
            # print(policy,Q[int(next_state_)][:])
            Q[int(state)][int(action)] = Q[int(state)][int(action)] + alpha*(reward_ + gamma*np.dot(policy,Q[int(next_state_)][:]) - Q[int(state)][int(action)])
            state = next_state_
            t_ = t_ + 1
        t.append(t_)

    return t

def Q_learning(reward,next_states,states,Q,moves):
    t=[]
    t_=0
    for episode in range(n_episode):
        state = states.index(start)
        while state != states.index(end):
            action = action_sel(Q,state,moves)
            reward_ = reward[int(state)][int(action)]
            next_state_ = next_states[int(state)][int(action)]
            Q[int(state)][int(action)] = Q[int(state)][int(action)] + alpha*(reward_ + gamma*np.max(Q[int(next_state_)][:]) - Q[int(state)][int(action)])
            state = next_state_
            t_ = t_ + 1
        t.append(t_)

    return t

def path(states,Q,next_states):
    path = [states.index(start)]
    state = states.index(start)
    while state != states.index(end):
        action = np.argmax(Q[int(state)][:])
        state = next_states[int(state)][int(action)]
        path.append(state)
    return path
        
t1=np.zeros(n_episode)
for i in range(10):
    np.random.seed(i)
    reward,next_states,states,Q=grid(moves,wind)
    t= Sarsa(reward,next_states,states,Q,moves)
    t1 = t1 + np.array(t)/10
state_value_plot(Q,"State-Value for Sarsa with Normal moves")
print("Path for Sarsa with Normal moves : ",path(states,Q,next_states))

t2=np.zeros(n_episode)
for i in range(10):
    np.random.seed(i)
    reward,next_states,states,Q=grid(kingmoves,wind)
    t= Sarsa(reward,next_states,states,Q,kingmoves)
    t2 = t2 + np.array(t)/10
state_value_plot(Q,"State-Value for Sarsa with Kingmoves")
print("Path for Sarsa with Kingmoves : ",path(states,Q,next_states))

t3=np.zeros(n_episode)
for i in range(10):
    np.random.seed(i)
    wind_ = np.array(wind) + np.array([np.random.choice([1,0,-1]) for i in range(n_col)])
    reward,next_states,states,Q=grid(kingmoves,wind_)
    t= Sarsa(reward,next_states,states,Q,kingmoves)
    t3 = t3 + np.array(t)/10
state_value_plot(Q,"State-Value for Sarsa with Kingmoves and Stochastic Winds")
print("Path for Sarsa with Kingmoves and Stochastic Winds : ",path(states,Q,next_states))

t4=np.zeros(n_episode)
for i in range(10):
    np.random.seed(i)
    reward,next_states,states,Q=grid(moves,wind)
    t= Exp_Sarsa(reward,next_states,states,Q,moves)
    t4 = t4 + np.array(t)/10
state_value_plot(Q,"State-Value for Expected Sarsa with Normal Moves")
print("Path for Expected Sarsa with Normal Moves : ",path(states,Q,next_states))

t5=np.zeros(n_episode)
for i in range(10):
    np.random.seed(i)
    reward,next_states,states,Q=grid(moves,wind)
    t= Q_learning(reward,next_states,states,Q,moves)
    t5 = t5 + np.array(t)/10
state_value_plot(Q,"State-Value for Q-learning with Normal Moves")
print("Path for Q-learning with Normal Moves : ",path(states,Q,next_states))

fig2 = plt.figure(figsize=(8,8))
plt.plot(t1,range(len(t1)),label='Normal Moves')
plt.plot(t2,range(len(t2)),label='King Moves')
plt.plot(t3,range(len(t3)),label='King Moves with Stochastic Winds')
plt.plot(t4,range(len(t4)),label='Normal Moves with Expected Sarsa')
plt.plot(t5,range(len(t5)),label='Normal Moves with Q-Learning')

plt.legend()
plt.savefig("Episode-time.png")