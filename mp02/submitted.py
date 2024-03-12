'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    
    frequency = {}
    for train_class in train:
        frequency[train_class] = Counter()
        for text in train[train_class]:
            for token in text:
                frequency[train_class][token] += 1

    # raise RuntimeError("You need to write this part!")
    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    nonstop = {}
    for class_type in frequency:
        nonstop[class_type] = Counter(frequency[class_type])
 
    for class_type in frequency:
        for text in frequency[class_type]:
            if text in stopwords:
                del nonstop[class_type][text]

    # raise RuntimeError("You need to write this part!")
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    # raise RuntimeError("You need to write this part!")
    likelihood = {}
    denominator = {}
    word_type = {}
    
    # finding the total word_type
    for temp_class_item in nonstop:
        word_type[temp_class_item] = len(set(nonstop[temp_class_item].keys()))
    
    # compute the denominator for each class
    for temp_class in nonstop:
        denominator[temp_class] = sum(nonstop[temp_class].values()) + smoothness*(word_type[temp_class]+1)         
    
    # finding P(token = X | Class = y)
    for temp_class in nonstop:
        likelihood[temp_class] = {} # creating a dict
        for  words in nonstop[temp_class]: # filling the dict
            likelihood[temp_class][words] = (nonstop[temp_class][words] + smoothness) / denominator[temp_class]

    # token OVV, P(toke = OVV | class = y)
    for temp_class in nonstop:
        likelihood[temp_class]['OOV'] = smoothness / denominator[temp_class]

    return likelihood
    

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = [''] * len(texts)
    # raise RuntimeError("You need to write this part!")

    # select the text
    for i in range(len(texts)):
        prob_pos = np.log(prior * len(texts))
        prob_neg = np.log((1.0-prior) * len(texts))
        for review_word in texts[i]:
            if review_word in stopwords: continue
            prob_pos += np.log(likelihood['pos'][review_word] if review_word in likelihood['pos'].keys() else likelihood['pos']['OOV'])
            prob_neg += np.log(likelihood['neg'][review_word] if review_word in likelihood['neg'].keys() else likelihood['neg']['OOV'])
        
        if prob_pos > prob_neg:
            hypotheses[i] = 'pos'
        elif prob_pos < prob_neg:  
            hypotheses[i] = 'neg'
        else:
            hypotheses[i] = 'undecided'

    return hypotheses

def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''

    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for i in range(len(priors)):
        for j in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop, smoothnesses[j])
            hypotheses = naive_bayes(texts, likelihood, priors[i])
            match_cnt = 0
            for (actual, guess) in zip(labels, hypotheses):
                if actual == guess: match_cnt += 1
            
            accuracies[i][j] = match_cnt / len(labels)

    # raise RuntimeError("You need to write this part!")

    return accuracies
                          
