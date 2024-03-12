'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

import pprint

# define your epsilon for laplace smoothing here

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!") 

    # for each word in the sentence: pos 0 is the word, pos 1 is the tag
    
    word_dict = dict(Counter()) # a dict of a dict, each word is a key, and compute the frequency of a certain tag, and pick the max
    tag_count = Counter()
    
    for sentence in train:
        for word in sentence:
            if word[0] not in word_dict:
                word_dict[word[0]] = Counter()
            word_dict[word[0]][word[1]] += 1
            tag_count[word[1]] += 1

    for word in word_dict:
        word_dict[word] = max(word_dict[word], key=word_dict[word].get)

    unseen = max(tag_count, key=tag_count.get) # get the word tag that is used most often
    test_results = []

    for sentence in test:
        new_sentence = []
        for word in sentence:
            if word not in word_dict:
                new_sentence.append((word, unseen)) # if the word is not in the training data, use the most frequent tag
            else:
                new_sentence.append((word, word_dict[word])) # if the word does, then use the word tag

        test_results.append(new_sentence) # append the sentence to the final result
                
    return test_results


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")
    
    # raw counts, without smoothing, for laplace smoothing, maybe use k = 1, for the smoothing
    tag_counter = Counter() 
    word_tags = dict() # each word may have multiple tags
    tag_pairs = dict() # key is previous, current is the other key
    tag_pairs['START'] = Counter()
    
    # counting process
    for sentence in train:
        for i in range(len(sentence)):   
            # tag_pairs[prev_to_curr] += 1 # count the number of times the tag pair appears
            if i > 0:
                if sentence[i-1][1] not in tag_pairs:
                    tag_pairs[sentence[i-1][1]] = Counter()
                tag_pairs[sentence[i-1][1]][sentence[i][1]] += 1

            tag_counter[sentence[i][1]] += 1 # count the number of times the tag appears
            
            if sentence[i][0] not in word_tags:
                word_tags[sentence[i][0]] = Counter()
            word_tags[sentence[i][0]][sentence[i][1]] += 1 # count the number of times the tag appears for a specific word
        
    
    # raw counts
    # pprint.pprint(tag_counter)
    # pprint.pprint(tag_pairs)
    # pprint.pprint(word_tags)

    # laplace smoothing probability and log
    k = 1e-5 
    transition = {key: {'UNKNOWN': k} for key in tag_pairs} # key is pair of tag (prev_tag, curr_tag), value is log(P(curr_tag|prev_tag))
    word_given_tag = {key: {'UNKNOWN': k} for key in tag_counter} # key is a tag, value is a dict of word to prob, emissions

    # pprint.pprint(transition)

    # emission
    # for a fixed T, the count the word is W --> P(word = W | tag = T)
    for words in word_tags:
        for tag in word_tags[words]:
            word_given_tag[tag][words] = word_tags[words][tag] + k    # adding laplace smoothing constant to each count
            # total_tag_count += word_given_tag[tag][words] method 1
            
    for tag in word_given_tag:
        tag_sum = sum(word_given_tag[tag].values())
        # tag_ratio = tag_sum / total_tag_count  # method 1 
        for word in word_given_tag[tag]:
            # word_given_tag[tag][word] = log(word_given_tag[tag][word]/total_tag_count)-log(tag_ratio) # method 1
            word_given_tag[tag][word] = log(word_given_tag[tag][word]/tag_sum) # method 2

    # pprint.pprint(word_given_tag)
    
    
    # P(Y_t = j | Y_(t-1) = i)
    # laplace smoothing prob and log transition probability
    for tag_prev in tag_pairs:
        for tag_curr in tag_pairs[tag_prev]:
            transition[tag_prev][tag_curr] = tag_pairs[tag_prev][tag_curr] + k
            # total_pair_count += transition[tag_prev][tag_curr] # method 1
    
    for tag in transition:
        tag_sum = sum(transition[tag].values()) # k + sum( k + Count(Y_t = j , Y_(t-1)=i))
        # tag_ratio = tag_sum / total_pair_count # method 1
        for next_tag in transition[tag]:
            # transition[tag][next_tag] = log(transition[tag][next_tag]/total_pair_count)-log(tag_ratio) # method 1
            transition[tag][next_tag] = log(transition[tag][next_tag]/tag_sum) # method 2
    
    # print(word_given_tag.keys())
    # pprint.pprint(word_given_tag)
    # pprint.pprint(transition)

    arr = list(word_given_tag.keys())
    index_to_tag = {i: arr[i] for i in range(len(arr))}   # each tag will have an index
    final_list = []
    
    # pprint.pprint(word_given_tag.keys())
    # return []

    for each_s in test:
        vb = np.zeros((len(arr), len(each_s)))
        backpointer = np.zeros((len(arr), len(each_s)), dtype=np.int32)
        for state in range(len(arr)):
            tag = index_to_tag[state]
            if each_s[0] in word_given_tag[tag]:    
                if tag in transition['START']: 
                    # vb[state][0] =  transition['START'][tag] + word_given_tag[tag][each_s[0]] # if there is one
                    vb[state][0] =  transition['START'][tag] # if there is one
                else:
                    # vb[state][0] =  transition['START']['UNKNOWN'] + word_given_tag[tag][each_s[0]] # if there is one
                    vb[state][0] =  transition['START']['UNKNOWN'] # if there is one
            else:
                if tag in transition['START']: 
                    # vb[state][0] = transition['START'][tag] + word_given_tag[tag]['UNKNOWN'] # if the word is never seen in with this tag/state
                    vb[state][0] = transition['START'][tag]  # if the word is never seen in with this tag/state
                else:
                    # vb[state][0] = transition['START']['UNKNOWN'] + word_given_tag[tag]['UNKNOWN']
                    vb[state][0] = transition['START']['UNKNOWN'] 
            backpointer[state][0] = -1 # all begins with start, mark this to something else for variable checking

        for t in range(1, len(each_s)):  # index for each step
            for s in range(len(arr)):   # index for each tag / state
                state_j = index_to_tag[s] # getting current state
                vb[s][t] = -math.inf
                for i in range(len(arr)):
                    state_i = index_to_tag[i] # grabbing the previous one
                    temp = 0
                    if 'START' == state_i and t > 1: continue
                    if 'END' == state_i: continue
                    if state_j in transition[state_i]: 
                        temp = vb[i][t-1] + transition[state_i][state_j]
                    else:
                        temp = vb[i][t-1] + transition[state_i]['UNKNOWN']
                    
                    if each_s[t] in word_given_tag[state_j]: 
                        temp += word_given_tag[state_j][each_s[t]]
                    else:
                        temp += word_given_tag[state_j]['UNKNOWN']
                    
                    if temp >= vb[s][t]:
                        vb[s][t] = temp
                        backpointer[s][t] = i   # record the state

                    # if state_j in transition[state_i]:    # transition is i to j
                    #     if each_s[t] in word_given_tag[state_j]: 
                    #         temp = vb[i][t-1] + transition[state_i][state_j] + word_given_tag[state_j][each_s[t]]
                    #         if temp >= vb[s][t]:
                    #             vb[s][t] = temp         # record the max value
                    #             backpointer[s][t] = i   # record the state
                    #     else:
                    #         temp = vb[i][t-1] + transition[state_i][state_j] + word_given_tag[state_j]['UNKNOWN']
                    #         if temp >= vb[s][t]:
                    #             vb[s][t] = temp         # record the max value
                    #             backpointer[s][t] = i   # record the state
                    # else:
                    #     if each_s[t] in word_given_tag[state_j]: 
                    #         temp = vb[i][t-1] + transition[state_i]['UNKNOWN'] + word_given_tag[state_j][each_s[t]]
                    #         if temp >= vb[s][t]:
                    #             vb[s][t] = temp         # record the max value
                    #             backpointer[s][t] = i   # record the state
                    #     else:
                    #         temp = vb[i][t-1] + transition[state_i]['UNKNOWN'] + word_given_tag[state_j]['UNKNOWN']
                    #         if temp >= vb[s][t]:
                    #             vb[s][t] = temp         # record the max value
                    #             backpointer[s][t] = i   # record the state  
                    
        # implement bestpathprob
        bestpathprob = -math.inf
        bespathpointer = -1
        for state in range(len(arr)):
            if vb[state][-1] > bestpathprob:
                bestpathprob = vb[state][-1]
                bespathpointer = state
        bestpath = [bespathpointer]
        for i in reversed(range(1,len(each_s))):
            bestpath.insert(0, backpointer[bespathpointer][i])
            bespathpointer = backpointer[bespathpointer][i]

        # print(len(each_s))
        # print(len(bestpath))
        # print(bestpath)

        new_sentence = []
        for i in range(len(each_s)):
            new_sentence.append((each_s[i], index_to_tag[bestpath[i]]))
        final_list.append(new_sentence)
        
    return final_list


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")
    # raw counts, without smoothing, for laplace smoothing, maybe use k = 1, for the smoothing
    tag_counter = Counter() 
    word_tags = dict() # each word may have multiple tags
    tag_pairs = dict() # key is previous, current is the other key
    tag_pairs['START'] = Counter()
    
    # counting process
    for sentence in train:
        for i in range(len(sentence)):   
            # tag_pairs[prev_to_curr] += 1 # count the number of times the tag pair appears
            if i > 0:
                if sentence[i-1][1] not in tag_pairs:
                    tag_pairs[sentence[i-1][1]] = Counter()
                tag_pairs[sentence[i-1][1]][sentence[i][1]] += 1

            tag_counter[sentence[i][1]] += 1 # count the number of times the tag appears
            
            if sentence[i][0] not in word_tags:
                word_tags[sentence[i][0]] = Counter()
            word_tags[sentence[i][0]][sentence[i][1]] += 1 # count the number of times the tag appears for a specific word
        
    
    # raw counts
    # pprint.pprint(tag_counter)
    # pprint.pprint(tag_pairs)
    # pprint.pprint(word_tags)

    # laplace smoothing probability and log

    hapax = {key: 1 for key in tag_counter}
    # filling hapax
    for word in word_tags:
        for tag in word_tags[word]:
            if word_tags[word][tag] == 1:
                hapax[tag] += 1

    hapax_sum = sum(hapax.values())
    for tag in hapax:
        hapax[tag] = hapax[tag] / hapax_sum
    

    # pprint.pprint(hapax)
    # return []

    k = 1e-5    
    transition = {key: {'UNKNOWN': k} for key in tag_pairs} # key is pair of tag (prev_tag, curr_tag), value is log(P(curr_tag|prev_tag))
    word_given_tag = {key: {'UNKNOWN': k * hapax[key]} for key in tag_counter} # key is a tag, value is a dict of word to prob, emissions
    
    # emission
    # for a fixed T, the count the word is W --> P(word = W | tag = T)
    for words in word_tags:
        for tag in word_tags[words]:
            word_given_tag[tag][words] = word_tags[words][tag] + word_given_tag[tag]['UNKNOWN']    # adding laplace smoothing constant to each count
            # total_tag_count += word_given_tag[tag][words] method 1
            
    for tag in word_given_tag:
        tag_sum = sum(word_given_tag[tag].values())
        # tag_ratio = tag_sum / total_tag_count  # method 1 
        for word in word_given_tag[tag]:
            # word_given_tag[tag][word] = log(word_given_tag[tag][word]/total_tag_count)-log(tag_ratio) # method 1
            word_given_tag[tag][word] = log(word_given_tag[tag][word]/tag_sum) # method 2

    # pprint.pprint(word_given_tag)

    # P(Y_t = j | Y_(t-1) = i)
    # laplace smoothing prob and log transition probability
    for tag_prev in tag_pairs:
        for tag_curr in tag_pairs[tag_prev]:
            transition[tag_prev][tag_curr] = tag_pairs[tag_prev][tag_curr] + k
            # total_pair_count += transition[tag_prev][tag_curr] # method 1
    
    for tag in transition:
        tag_sum = sum(transition[tag].values()) # k + sum( k + Count(Y_t = j , Y_(t-1)=i))
        # tag_ratio = tag_sum / total_pair_count # method 1
        for next_tag in transition[tag]:
            # transition[tag][next_tag] = log(transition[tag][next_tag]/total_pair_count)-log(tag_ratio) # method 1
            transition[tag][next_tag] = log(transition[tag][next_tag]/tag_sum) # method 2
    
    # print(word_given_tag.keys())
    # pprint.pprint(word_given_tag)
    # pprint.pprint(transition)

    arr = list(word_given_tag.keys())
    index_to_tag = {i: arr[i] for i in range(len(arr))}   # each tag will have an index
    final_list = []
    
    # pprint.pprint(word_given_tag.keys())
    # return []

    for each_s in test:
        vb = np.zeros((len(arr), len(each_s)))
        backpointer = np.zeros((len(arr), len(each_s)), dtype=np.int32)
        for state in range(len(arr)):
            tag = index_to_tag[state]
            if each_s[0] in word_given_tag[tag]:    
                if tag in transition['START']: 
                    # vb[state][0] =  transition['START'][tag] + word_given_tag[tag][each_s[0]] # if there is one
                    vb[state][0] =  transition['START'][tag] # if there is one
                else:
                    # vb[state][0] =  transition['START']['UNKNOWN'] + word_given_tag[tag][each_s[0]] # if there is one
                    vb[state][0] =  transition['START']['UNKNOWN']# if there is one

            else:
                if tag in transition['START']: 
                    # vb[state][0] = transition['START'][tag] + word_given_tag[tag]['UNKNOWN'] # if the word is never seen in with this tag/state
                    vb[state][0] = transition['START'][tag]  # if the word is never seen in with this tag/state

                else:
                    # vb[state][0] = transition['START']['UNKNOWN'] + word_given_tag[tag]['UNKNOWN']
                    vb[state][0] = transition['START']['UNKNOWN']

            backpointer[state][0] = -1 # all begins with start, mark this to something else for variable checking

        for t in range(1, len(each_s)):  # index for each step
            for s in range(len(arr)):   # index for each tag / state
                state_j = index_to_tag[s] # getting current state
                vb[s][t] = -math.inf
                for i in range(len(arr)):
                    state_i = index_to_tag[i] # grabbing the previous one
                    if 'START' == state_i and t > 1: continue
                    if 'END' == state_i: continue
                    if state_j in transition[state_i]:    # transition is i to j
                        if each_s[t] in word_given_tag[state_j]: 
                            temp = vb[i][t-1] + transition[state_i][state_j] + word_given_tag[state_j][each_s[t]]
                            if temp >= vb[s][t]:
                                vb[s][t] = temp         # record the max value
                                backpointer[s][t] = i   # record the state
                        else:
                            temp = vb[i][t-1] + transition[state_i][state_j] + word_given_tag[state_j]['UNKNOWN']
                            if temp >= vb[s][t]:
                                vb[s][t] = temp         # record the max value
                                backpointer[s][t] = i   # record the state
                    else:
                        if each_s[t] in word_given_tag[state_j]: 
                            temp = vb[i][t-1] + transition[state_i]['UNKNOWN'] + word_given_tag[state_j][each_s[t]]
                            if temp >= vb[s][t]:
                                vb[s][t] = temp         # record the max value
                                backpointer[s][t] = i   # record the state
                        else:
                            temp = vb[i][t-1] + transition[state_i]['UNKNOWN'] + word_given_tag[state_j]['UNKNOWN']
                            if temp >= vb[s][t]:
                                vb[s][t] = temp         # record the max value
                                backpointer[s][t] = i   # record the state  

        # implement bestpathprob
        bestpathprob = -math.inf
        bespathpointer = -1
        for state in range(len(arr)):
            if vb[state][-1] > bestpathprob:
                bestpathprob = vb[state][-1]
                bespathpointer = state
        bestpath = [bespathpointer]
        for i in reversed(range(1,len(each_s))):
            bestpath.insert(0, backpointer[bespathpointer][i])
            bespathpointer = backpointer[bespathpointer][i]

        # print(len(each_s))
        # print(len(bestpath))
        # print(bestpath)

        new_sentence = []
        for i in range(len(each_s)):
            new_sentence.append((each_s[i], index_to_tag[bestpath[i]]))
        final_list.append(new_sentence)
        
    return final_list


