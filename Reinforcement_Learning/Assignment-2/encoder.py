import argparse,copy,pulp
import numpy as np

def cmd_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid')
    args=parser.parse_args()

    pathname = args.grid

    f = open(pathname)
    content=f.readlines()
    n_state=0
    states=[]

    for i,line in enumerate(content):
        row = line[:-1].split(" ")
        for j,element in enumerate(row):
            if element != '1':
                states.append((i,j))
                n_state += 1
                if element == '2':
                    start = n_state-1
                if element == '3':
                    end = n_state-1
    transition=[]
    for state in states:
        adj_states = [(state[0]-1,state[1]),(state[0],state[1]+1),(state[0]+1,state[1]),(state[0],state[1]-1)]
        # N E S W
        r1 = -1
        r2 = 50
        r3 = -1
        if state == states[end]:
            continue

        for action in range(4): 
            adj_state = adj_states[action]
            if adj_state in states:
                if adj_state == states[end]:
                    transition.append([states.index(state),action,end,r2,1])
                else:
                    transition.append([states.index(state),action,states.index(adj_state),r1,1])
            else:
                transition.append([states.index(state),action,states.index(state),r3,1])

    discount = 1
    mdptype = 'episodic'
                
    return n_state,4,start,end,transition,mdptype,discount

n_state,n_action,start,end,transition,mdptype,discount = cmd_argparser()

print('numStates %s'%(str(n_state)))
print('numActions %s'%(str(n_action)))
print('start %s'%(str(start)))
print('end %s'%(str(end)))
for trans in transition:
    print('transition %s %s %s %s %s'%(str(trans[0]),str(trans[1]),str(trans[2]),str(trans[3]),str(trans[4])))
print('mdptype %s'%(mdptype))
print('discount  %s'%(discount))
