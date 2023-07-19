import argparse,copy,pulp
import numpy as np

def cmd_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp')
    parser.add_argument('--algorithm')
    args=parser.parse_args()

    pathname = args.mdp
    algo = args.algorithm

    f = open(pathname)
    content=f.readlines()
    for i in range(len(content)):
        line = content[i].split(" ")
        if line[0] == 'numStates':
            n_state = int(line[1])
        elif line[0] == 'numActions':
            n_action = int(line[1])
            transition=[]
            for state in range(n_state):
                row=[]
                for action in range(n_action):
                    col=[[0,0,0]]
                    row.append(col)
                transition.append(row)

        elif line[0] == 'start':
            start = int(line[1])
        elif line[0] == 'end':
            end_states=[]
            for j in range(len(line)-1):
                end_states.append(int(line[j+1]))
        elif line[0] == 'mdptype':
            mdptype = line[1][:-1]
        elif line[0] == 'discount':
            discount = float(line[2])
        elif line[0] == 'transition':
            # Transition[state][action]=[nstate,reward,probablity]
            transition[int(line[1])][int(line[2])].append([int(line[3]),float(line[4]),float(line[5])])
            
    return n_state,n_action,start,end_states,mdptype,discount,algo,transition

def value_iter(n_state,n_action,delta,discount,transition):
    V = np.zeros(n_state)
    policy = np.random.randint(0,n_action,n_state)
    error = 1
    while error > delta :
        Vcopy = copy.deepcopy(V)
        for state in range(n_state):
            values=[]
            for action in range(n_action):
                trans=transition[state][action]
                value=0
                if len(trans) != 1:
                    for option in trans[1:]:
                        value = value + option[2]*(option[1]+discount*V[option[0]])
                values.append(value)
            V[state] = np.max(values)
            policy[state] = np.argmax(values)
        error = np.max(np.linalg.norm(V-Vcopy),0)

    return V,policy


def policy_iter(n_state,n_action,delta,discount,transition):
    stable = False
    policy = np.zeros(n_state, dtype=int)
    while not stable  :
        V = np.zeros(n_state)
        error = 1
        while error > delta:
            Vcopy = copy.deepcopy(V)
            for state in range(n_state):
                action=policy[state]
                trans=transition[state][action]
                value = 0
                if len(trans) != 1:
                    for option in trans[1:]:
                        value = value + option[2]*(option[1]+discount*V[option[0]])
                    V[state]=value
            error = np.max(np.linalg.norm(V-Vcopy),0)

        stable = True
        for state in range(n_state):
            values = []
            action = policy[state]
            for action_ in range(n_action):
                trans = transition[state][action_]
                value = 0
                if len(trans) != 1:
                    for option in trans[1:]:
                        value = value + option[2]*(option[1]+discount*V[option[0]])
                values.append(value)
                policy[state] = np.argmax(values)
            if action != policy[state]:
                stable = False 

    return V,policy   

def LinearProg(n_state,n_action,discount,transition):
    model = pulp.LpProblem("MDP",pulp.LpMinimize)
    Des_Var = []
    for state in range(n_state):
        variable = pulp.LpVariable("{0:0=7d}".format(state))
        Des_Var.append(variable)
    o=0
    for state in range(n_state):
        o += Des_Var[state]
    model += o

    for state in range(n_state):
        c=0
        for action in range(n_action):
            trans=transition[state][action]
            value=0
            if len(trans) != 1:
                for option in trans[1:]:
                    value = value + option[2]*(option[1]+discount*Des_Var[option[0]])
            c = value
            model += Des_Var[state] >= c
    model.writeLP('lin.lp')
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    V = np.zeros(n_state)

    for i,var in enumerate(model.variables()):
        V[i] = var.varValue

    policy = np.zeros(n_state)
    for state in range(n_state):
        values=[]
        for action in range(n_action):
            trans=transition[state][action]
            value=0
            if len(trans) != 1:
                for option in trans[1:]:
                    value = value + option[2]*(option[1]+discount*V[option[0]])
            values.append(value)
        policy[state] = np.argmax(values)

    return V,policy



n_state,n_action,start,end_states,mdptype,discount,algo,transition=cmd_argparser()
delta=1e-12

if algo == 'vi' :
    Value_func,policy = value_iter(n_state,n_action,delta,discount,transition)
if algo == 'hpi' :
    Value_func,policy = policy_iter(n_state,n_action,delta,discount,transition)
if algo == 'lp' :
    Value_func,policy = LinearProg(n_state,n_action,discount,transition)    

for state in range(n_state):
    text = "{:.6f} ".format((Value_func[state]))+"{:.0f}".format((policy[state]))
    print(text)


