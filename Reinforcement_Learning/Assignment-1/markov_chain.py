import numpy as np
n = int(input("Number of States : "))
transition_matrix = []
for i in range(n):
    row=[]
    for j in range(n):
        row.append(float(input("Enter the probabity of going to State %s to State %s : "%(i,j))))
    transition_matrix.append(row)
transition_arr=np.array(transition_matrix)
Start_state=int(input("starting state index is :"))
reaching_state=int(input("reaching state index is :"))
t= int(input('Time :'))
for i in range(t):
    transition_arr=transition_arr.dot(np.array(transition_matrix))

print("The probablity of reaching State %s from State %s at Time = %s is :"%(Start_state,reaching_state,t),transition_arr[Start_state][reaching_state])