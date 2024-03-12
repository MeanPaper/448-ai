# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

from queue import PriorityQueue
from queue import Queue


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    start = maze.start  # the start of the maze
    dest = None
    # use to store a set of tuple, tuple defined as (row, col), 
    path = []       # recording the path coodinate
    parents = dict()    # parent and explored set
    frontier = Queue()  # this is a queue using array, use append to do push, use pop(0) to pop fromt the queue
    frontier.put(start)
    parents[start] = (-1,-1) # start has no parents  
    while(not frontier.empty()):    # see if frontier is empty
        cur_pos = frontier.get()
        if(maze[cur_pos[0], cur_pos[1]] == maze.legend.waypoint): 
            dest = cur_pos
            break
        for near in maze.neighbors(cur_pos[0], cur_pos[1]):
            if near in parents: continue  # see if the neighbor has been visited or not
            parents[near] = cur_pos     # record the previous node
            frontier.put(near)       # push the neighbor to frontier queue  
        
    # print(parents)    
    while(dest != None and dest != (-1,-1)):
        path.insert(0, dest)
        dest = parents[dest]
    
    return path

def manhatton(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """


    #TODO: Implement astar_single
    
    # a helper function for finding the heuristic
    previous = dict()     # the previous node of the current node
    cost = dict()
    frontier = PriorityQueue()
    path = []
    start = maze.start  # start of the matrix
    dest = maze.waypoints[0]
    previous[start] = (-1,-1) # start has no previous node
    cost[start] = 0     # start begins with 0 cost

    frontier.put((0,start))
    # frontier.put((2,(1,2)))
    # frontier.put((1,(3,4)))

    # print(frontier.get())
    # print(frontier.get())
    # print(frontier.get())
    while(not frontier.empty()):
        current = frontier.get()[1]    # getting the node from the queue
        if maze[current[0], current[1]] == maze.legend.waypoint:
            break
        for near in maze.neighbors(current[0], current[1]):
            update_cost = cost[current] + 1 # because it is an maze, so the cost=1 per step
            if near not in cost or update_cost < cost[near]:
                cost[near] = update_cost
                priority = cost[near] + manhatton(dest, near) # find the priority
                frontier.put((priority, near))  # push back to the priority queue
                previous[near] = current

    # print(previous)
    # print(dest)
    while(dest != (-1,-1)):
        path.insert(0,dest)
        dest = previous[dest]

    return path



def construct_MST(dest_set):
    
    if len(dest_set) == 0:
        return 0

    # graph contruction
    adjmatrix = dict()
    for i in range(len(dest_set)):
        adjmatrix[dest_set[i]] = []
        for j in range(0 ,len(dest_set)):
            adjmatrix[dest_set[i]].append(manhatton(dest_set[i], dest_set[j]))

    # prims algo for mst
    key = [1000000000]*len(dest_set)
    previous = dict()
    key[0] = 0
    mst = [False] * len(dest_set)
    previous[dest_set[0]] = (-1,-1)
    for node in range(len(dest_set)):
        mini = 1000000000
        min_idx = -1
        for v in range(len(dest_set)):
            if key[v] < mini and mst[v] == False:
                mini = key[v]
                min_idx = v
    
        mst[min_idx] = True
    
        for v in range(len(dest_set)):
            if adjmatrix[dest_set[min_idx]][v] > 0 and mst[v] == False and key[v] > adjmatrix[dest_set[min_idx]][v]:
                key[v] = adjmatrix[dest_set[min_idx]][v]
                previous[dest_set[v]] = dest_set[min_idx]
    
    # sum all the edge
    sum = 0
    for points in previous.keys():
        if previous[points] == (-1,-1):
            sum += 0
        else:
            sum += manhatton(points, previous[points])
    return sum
    
# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    path = [] 
    start = maze.start
    dest_dict = dict()
    dp_dict = dict()
    dest_list = list(maze.waypoints)
    dest_seen = tuple([False] * len(dest_list))
    
    for i in range(len(dest_list)):
        dest_dict[dest_list[i]] = i # map destinations to specific index

    pq = PriorityQueue()            # store tuple base on the f score priority
    pq.put((0,0,start, dest_seen))  # (f_score, g_score, start, dest_setup)
    dp_dict[(start, dest_seen)] = (0,0, None) # use to keep track of the path
    
    # the end
    terminate = (-1,-1)                             
    
    while(not pq.empty()):
        if terminate != (-1,-1):
            break

        # pop content from the top of min heap
        pq_top = pq.get()
        g_score = pq_top[1]
        point = pq_top[2]
        dest_visit = pq_top[3]

        # grab neighbor
        for near in maze.neighbors(point[0],point[1]):
            new_dest_visit = dest_visit # track new state
            new_g_score = g_score + 1   # get new cost

            if near in dest_list and new_dest_visit[dest_dict[near]] == False:  # mark dest node
                new_dest_visit = list(new_dest_visit)
                new_dest_visit[dest_dict[near]] = True
                new_dest_visit = tuple(new_dest_visit)


            # heuristic calculation and MST
            new_h_score = 1000000000
            cnt = 0
            mst_set = []
            for j in range(len(dest_list)):
                if new_dest_visit[j] == False:
                    new_h_score = min(new_h_score, manhatton(near, dest_list[j]))
                    mst_set.append(dest_list[j])
                else:
                    cnt += 1
            if cnt == len(dest_list):
                new_h_score = 0
            new_h_score += construct_MST(mst_set)
            
            # update node conditions and see if all dest have been visited
            new_pq_tuple = (new_g_score + new_h_score, new_g_score, near, new_dest_visit)
            if (near, new_dest_visit) not in dp_dict or dp_dict[near,new_dest_visit][0] > new_pq_tuple[0]:
                dp_dict[(near, new_dest_visit)] = (new_pq_tuple[0], new_pq_tuple[1], (point, dest_visit)) # record the path
                pq.put(new_pq_tuple)
                if cnt == len(dest_list):
                    terminate = (near, new_dest_visit)
                    break

    # path printing
    path = [terminate[0]]
    while (0 != dp_dict[terminate][1]):
        terminate = dp_dict[terminate][2]
        path.insert(0, terminate[0])

    for each in dp_dict:
        print(each)
        print(dp_dict[each])
    return path