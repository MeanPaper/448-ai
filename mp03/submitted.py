'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - 1-D array of k images, the k nearest neighbors of image
    labels - 1-D array of k labels corresponding to the k images
    '''


    # raise RuntimeError('You need to write this part!')

    neighbors = np.zeros((k, len(image)))  # empty array with length k
    labels = np.zeros((k), dtype=bool) # empty array with k labels
    euclidean_distance = []
    
    # never train_image[0] is the image
    for i in range(len(train_images)):
        temp = [np.linalg.norm(image - train_images[i]), i]
        euclidean_distance.append(temp)
    euclidean_distance.sort() # sort by euclidean distance


    for i in range(k):
        index = euclidean_distance[i][1]
        temp = train_images[index]
        # neighbor= temp
        neighbors[i] = temp
        labels[i] = train_labels[index]
        

    # print(euclidean_distance)
    
    return neighbors, labels

    
    
    



def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    
    # raise RuntimeError('You need to write this part!')
    hypotheses = []
    scores = [] 

    for dev_img in dev_images:
        knn_neighbor, vote_label = k_nearest_neighbors(dev_img, train_images, train_labels, k)
        true_cnt = (list)(vote_label).count(True)
        false_cnt = (list)(vote_label).count(False)
        if true_cnt > false_cnt: 
            hypotheses.append(1)
            scores.append(true_cnt)
        else:
            hypotheses.append(0)
            scores.append(false_cnt)


    # test = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 
    #         0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 
    #         1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 
    #         1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 
    #         1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 
    #         1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 
    #         1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 
    #         1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 
    #         1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 
    #         1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 
    #         1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    # wrong = 0
    # for (h, t) in zip(hypotheses, test):
    #     if(h != t): wrong += 1
    # print("Wrong count: ", wrong)

    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''

    # raise RuntimeError('You need to write this part!')
    confusions = np.zeros((2,2), dtype=int)
    accuracy = 0.0
    f1 = 0.0
    
    for (h, r) in zip(hypotheses, references):
        confusions[(int)(r)][(int)(h)] += 1 

    precision = (float)(confusions[1,1]) / (confusions[0,1] + confusions[1,1]) 
    recall = (float)(confusions[1,1]) / (confusions[1,1] + confusions[1,0])
    accuracy = (float)(confusions[1,1] + confusions[0,0]) / (sum(confusions[0]) + sum(confusions[1]))
    f1 = 2 / ((1/recall)+(1/precision))


    return confusions, accuracy, f1
