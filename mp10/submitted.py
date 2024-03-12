'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3


directions = ["left", "up", "right", "down"]
direct_dict = {"left":[0,-1], "up":[-1,0], "right":[0,1], "down":[1,0]}
clockwise = {"left": "up", "down": "left", "up": "right", "right": "down"}
counterclockwise = {"left": "down", "down": "right", "right": "up", "up":"left"}

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') 
    if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    # raise RuntimeError("You need to write this part!")


    # model.W, the wall matrix
    # model.D, the direction probability matrix, same direction [r,c,0], counter-clock right angle [r,c,1], clock right angle [r,c,2]
    # model.T  the terminal state matrix

    M = model.M
    N = model.N
    P = np.zeros((M, N, 4, M, N))

    for r in range(M):      # row counter
        for c in range(N):  # column counter
            if(model.T[r,c] == True): continue
            for a in range(4):
                dir = directions[a]
                new_r, new_c = r + direct_dict[dir][0], c + direct_dict[dir][1] # final out the next direction

                if(new_r >= 0 and new_c >= 0 and new_r < M and new_c < N):      # bounds checking
                    if model.W[new_r, new_c] == True:                           # move in the right direction but there is a wall
                        P[r,c,a,r,c] +=  model.D[r,c,0] 
                    else:
                        P[r,c,a,new_r, new_c] += model.D[r,c,0]
                else:
                    P[r,c,a,r,c] += model.D[r,c,0]

                # clockwise, wrong direction
                clock_dir = direct_dict[clockwise[dir]]
                new_r, new_c = r + clock_dir[0], c + clock_dir[1]       
                if(new_r >= 0 and new_c >= 0 and new_r < M and new_c < N):  # bound checking for misleading dir
                    if model.W[new_r, new_c] == True:                       # move to the wall
                        P[r,c,a,r,c] +=  model.D[r,c,2] 
                    else:
                        P[r,c,a,new_r, new_c] += model.D[r,c,2]
                else:
                    P[r,c,a,r,c] += model.D[r,c,2]

                # counter-clockwise, wrong direction
                counter_clock_dir = direct_dict[counterclockwise[dir]]
                new_r, new_c = r + counter_clock_dir[0], c + counter_clock_dir[1]       
                if(new_r >= 0 and new_c >= 0 and new_r < M and new_c < N):  # bound checking for misleading dir
                    if model.W[new_r, new_c] == True:                       # move to the wall
                        P[r,c,a,r,c] +=  model.D[r,c,1] 
                    else:
                        P[r,c,a,new_r, new_c] += model.D[r,c,1]
                else:
                    P[r,c,a,r,c] += model.D[r,c,1]

    return P
    

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")
    gamma = model.gamma
    M = model.M 
    N = model.N
    U_next = np.zeros((M, N))

    # for each current row and current column
    for r in range(M):
        for c in range(N):
            if (model.T[r,c]==True):        # no change on the rewards
                U_next[r,c] = model.R[r,c]
                continue        
            for a in range(4):
                dir = directions[a]         # get the direction
                temp_value = 0
                new_r, new_c = r + direct_dict[dir][0], c + direct_dict[dir][1]  # right dir
                if(new_r >= 0 and new_c >= 0 and new_r < M and new_c < N):       # bounds checking
                    if model.W[new_r,new_c] == False:                           
                        temp_value += P[r,c,a,new_r,new_c]*U_current[new_r][new_c]
                
                # clockwise, wrong direction, we only consider the non wall cases
                clock_dir = direct_dict[clockwise[dir]]
                new_r, new_c = r + clock_dir[0], c + clock_dir[1]      
                if(new_r >= 0 and new_c >= 0 and new_r < M and new_c < N):  # bound checking for misleading dir
                    if model.W[new_r, new_c] == False:                       
                        temp_value += P[r,c,a,new_r,new_c]*U_current[new_r][new_c]
                
                # counter-clockwise, wrong direction, we only consider the non wall cases
                counter_clock_dir = direct_dict[counterclockwise[dir]]
                new_r, new_c = r + counter_clock_dir[0], c + counter_clock_dir[1]       
                if(new_r >= 0 and new_c >= 0 and new_r < M and new_c < N):  # bound checking for misleading dir
                    if model.W[new_r, new_c] == False:                       
                        temp_value += P[r,c,a,new_r,new_c]*U_current[new_r][new_c]

                if(P[r,c,a,r,c] != 0):  # if the probability is non zero, it means the robot stays here more than once
                                        # meaning that 3 conditions above might have 1 to 3 of them not fullfill
                    temp_value += P[r,c,a,r,c]*U_current[r][c]

                U_next[r, c] = max(U_next[r,c], temp_value) # user the max
            U_next[r,c] = U_next[r,c] * gamma + model.R[r,c]
    
    return U_next


def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition_matrix(model)
    current = np.zeros((model.M, model.N))
    next = update_utility(model, P, current)
    sum_current = np.sum(current)
    sum_next = np.sum(next)
    while(abs(sum_next - sum_current) >= epsilon):
        current = next
        next = update_utility(model, P, current)
        sum_current = np.sum(current)
        sum_next = np.sum(next)
    return next

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
