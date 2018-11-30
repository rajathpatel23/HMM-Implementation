# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 02:44:07 2018

@author: rajat.patel
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 23:16:37 2018

@author: rajat.patel
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict 

THRESHOLD = 1
LAMBDA_1 = 0.0

#DATA_LOCATION = "G:\\Application_2016\\Natural_Language_Processing\\data"
DATA_LOCATION = "C:\\Users\\rajat.patel\\Documents\\my_NLP\\data\\data"

data_read = open(DATA_LOCATION+"\\el_gdt-ud-train.conllu", "r", encoding="utf-8")
data_read_train = data_read.readlines()

def reading_data_token_types(data, mode=None):
    token_list = []
    UPOS_list = []
    XPOS_list = []
    q = 0
    for x in data:
        if x.startswith("#"):
            continue
        else:
            line = x.split("\t")
            if line[0] == "1":
                q+=1
                token_list.append("BOS")
                UPOS_list.append("BOS")
                XPOS_list.append("BOS")
            if len(line) == 1:
                token_list.append("EOS")
                UPOS_list.append("EOS")
                XPOS_list.append("EOS")
                continue
            else:
                f = line[1].lower()
                token_list.append(f)
                UPOS_list.append(line[3])
                XPOS_list.append(line[4])
    return(q, token_list, UPOS_list, XPOS_list)


def maximum_likelihood(word_counter, V):
    N = sum(word_counter.values())
    for words in word_counter.keys():
        word_counter[words] = (word_counter[words])/(N+V)
    return(word_counter)

def prior_probability(Counter_1):
    prior_values = defaultdict()
    total = sum(Counter_1.values())
    for key in Counter_1.keys():
        prior_values[key] = Counter_1[key]/total
    return(prior_values)
    
def introduce_OOV(token_list, threshold, old_counter):
#    token_with_OOV = []
    for i in range(len(token_list)):
        if old_counter[token_list[i]] <= threshold:
            token_list[i] = "OOV"
    return(token_list)
#%%  
def accuracy_sentence_level(true_label_1, predicted_values_dev_1, token_list, NOS):
    a = 0
    true_sentence_tag = []
    predicted_sentence_tag = []
    for i in range(len(predicted_values_dev_1)):
        true_sentence_tag.append(true_label_1[i])
        predicted_sentence_tag.append(predicted_values_dev_1[i])
        if token_list[i] == "EOS":
            true_sentence_tag = np.array(true_sentence_tag)
            predicted_sentence_tag = np.array(predicted_sentence_tag)
            if np.array_equal(true_sentence_tag, predicted_sentence_tag):
                a+=1
            true_sentence_tag = []
            predicted_sentence_tag = []        
    accuracy = a/NOS
    return(accuracy)
#%%
def give_true_label_list(token_label_dict, UPOS_list_dev):
    true_label = []
    for tag in UPOS_list_dev:
        true_label.append(token_label_dict[tag])
    return(true_label)
    

def give_tag_labels(unique_token_list_dev):
    token_label_dict = defaultdict()
    for i in range(len(unique_token_list_dev)):
        token_label_dict[unique_token_list_dev[i]] = i
    return(token_label_dict)
    

def transition_probability_matrix(UPOS_list, Counter_UPOS, unique_token_list, label_tags):
    dictionary_count = defaultdict(int)
    m = len(unique_token_list)
    transition_matrix = np.zeros((m, m))
    for i in range(len(UPOS_list)-1):
        dictionary_count[(UPOS_list[i], UPOS_list[i+1])] +=1

    for keys in dictionary_count.keys():
        transition_matrix[label_tags[keys[0]], label_tags[keys[1]]] = dictionary_count[(keys)]

    for i in range(len(transition_matrix)):
        c = np.sum(transition_matrix[i])
        transition_matrix[i] = transition_matrix[i]/c
    return(dictionary_count, transition_matrix)


def viterbi(initial_probability_1, transition_matrix, emission_probability, observations, token_with_OOV, token_types):
    M = len(observations)
    S = initial_probability_1.shape[0]
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    backpointers = np.zeros((M, S), 'int')
    #forward algorithm
    alpha[0, :] = initial_probability_1*emission_probability[:, token_with_OOV.index(observations[0])]
    for t in range(1,M):
        for s2 in range(S):
            for s1 in range(S):
                if observations[t] in token_types:
                    score = alpha[t-1][s1] * transition_matrix[s1][s2] * emission_probability[s2][token_with_OOV.index(observations[t])]
                else:
                    score = alpha[t-1][s1] * transition_matrix[s1][s2] * emission_probability[s2][token_with_OOV.index("OOV")]
                if score > alpha[t][s2]:
                    alpha[t][s2] = score
                    backpointers[t][s2] = s1
    sequence_list = []
    sequence_list.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        sequence_list.append(backpointers[i, sequence_list[-1]])
    final_sequence = np.array(list(reversed(sequence_list)))
    return final_sequence, np.max(alpha[M-1,:])

#%%
def build_store_probabilities(token_with_OOV, UPOS_list):
    store_probabilities = defaultdict(int)
    for k in range(len(token_with_OOV)):
        store_probabilities[(token_with_OOV[k], UPOS_list[k])]+=1
    return(store_probabilities)

def get_emission_matrix(store_probabilities, token_with_OOV, label_tags, Counter_UPOS):
    emission_matrix = np.zeros((len(token_with_OOV), len(label_tags)))
    for keys in store_probabilities:
        emission_matrix[token_with_OOV.index(keys[0]), label_tags[keys[1]]] = store_probabilities[keys]
    emission_matrix = emission_matrix.T
    for key in label_tags:
        emission_matrix[label_tags[key]] = emission_matrix[label_tags[key]]/(Counter_UPOS[key])
    return(emission_matrix)




def accuracy_word_level(true_label_1, predicted_values_dev_1):
    a = 0
    for i in range(len(predicted_values_dev_1)):
         if predicted_values_dev_1[i] == true_label_1[i]:
             a= a+1
    accuracy = a/len(predicted_values_dev_1)
    return(accuracy)

#%%
    
def main():
    Number_of_sentences, token_list , UPOS_list, XPOS_list = reading_data_token_types(data_read_train)
    print(Number_of_sentences)
    corpora_word_count = Counter(token_list)
    token_with_OOV = introduce_OOV(token_list, THRESHOLD, corpora_word_count)
    new_corpora_word_count = Counter(token_with_OOV)
    token_types = list(new_corpora_word_count.keys())
#    V = len(new_corpora_word_count)
    Counter_UPOS= dict(Counter(UPOS_list))
    Counter_XPOS = dict(Counter(XPOS_list))
#    Prior_UPOS = prior_probability(Counter_UPOS)
#    Prior_XPOS = prior_probability(Counter_XPOS)
    
    
    
    
    unique_token_list = list(Counter_UPOS.keys())
    label_tags = give_tag_labels(unique_token_list)
    store_probabilities = build_store_probabilities(token_with_OOV, UPOS_list)
    dictionary_count, transition_matrix = transition_probability_matrix(UPOS_list, Counter_UPOS, unique_token_list, label_tags)
    print(label_tags)
    emission_matrix = get_emission_matrix(store_probabilities, token_with_OOV, label_tags, Counter_UPOS)
    initial_probabiltiy = []
    for  i in range(len(label_tags)):
        initial_probabiltiy.append(1/(len(label_tags)))   #uniform distribution
    initial_probabiltiy = np.array(initial_probabiltiy)
    print(initial_probabiltiy)
    # print(emission_matrix[0])
    #%%
    data_dev = open(DATA_LOCATION+"\\el_gdt-ud-dev.conllu", 'r', encoding="utf-8")
    data_dev_read = data_dev.readlines()
    Number_of_sentences_dev, token_list_dev, UPOS_list_dev, XPOS_list_dev = reading_data_token_types(data_dev_read)
    sentences_dev = []
    temp_list = []
    for i in range(len(token_list_dev)):
        if token_list_dev[i] == "EOS":
            temp_list.append("EOS")
            sentences_dev.append(temp_list)
            temp_list = []
        else:
            temp_list.append(token_list_dev[i])
    print(len(sentences_dev))
    print(Number_of_sentences_dev)
    sentences_dev = np.array(sentences_dev)
    predicted_values_dev = []
    for u in range(len(sentences_dev)):
        observations_1 = sentences_dev[u]
        final_sequence, score = viterbi(initial_probabiltiy, transition_matrix, emission_matrix, observations_1, token_with_OOV, token_types)
        print(final_sequence)
        predicted_values_dev.append(final_sequence)
        
    predicted_values_dev = np.array(predicted_values_dev)
    print(predicted_values_dev.shape)
    predicted_values_dev = np.hstack(predicted_values_dev)
    true_label = give_true_label_list(label_tags,UPOS_list_dev)
    accuracy_word_dev = accuracy_word_level(true_label, predicted_values_dev)
    print("############## WORD LEVEL ACCURACY FOR UPOS TAGS ON DEV DATA ######################")
    print(accuracy_word_dev)
    accuracy_sentence_dev = accuracy_sentence_level(true_label, predicted_values_dev, token_list_dev, Number_of_sentences_dev)
    print("############## SENTENCE_LEVEL ACCURACY FOR UPOS TAGS ON DEV DATA ######################")
    print(accuracy_sentence_dev)
    
    #%%
    data_test = open(DATA_LOCATION+"\\el_gdt-ud-test.conllu", 'r', encoding="utf-8")
    data_test_read = data_test.readlines()
    Number_of_sentences_test, token_list_test, UPOS_list_test, XPOS_list_test = reading_data_token_types(data_test_read)
    sentences_test = []
    temp_list_test = []
    #%%
    for i in range(len(token_list_test)):
        if token_list_test[i] == "EOS":
            temp_list_test.append("EOS")
            sentences_test.append(temp_list_test)
            temp_list_test = []
        else:
            temp_list_test.append(token_list_test[i])
    print(len(sentences_test))
    print(Number_of_sentences_test)
    sentences_test = np.array(sentences_test)
    #%%
    predicted_values_test = []
    for u1 in range(len(sentences_test)):
        observations_1_test = sentences_test[u1]
        final_sequence_test, score = viterbi(initial_probabiltiy, transition_matrix, emission_matrix, observations_1_test, token_with_OOV, token_types)
    #    print(final_sequence_test)
        predicted_values_test.append(final_sequence_test)
        
    predicted_values_test = np.array(predicted_values_test)
    print(predicted_values_test.shape)
    predicted_values_test = np.hstack(predicted_values_test)
    true_label_test = give_true_label_list(label_tags,UPOS_list_test)
    accuracy_word_test = accuracy_word_level(true_label_test, predicted_values_test)
    print("############## WORD LEVEL ACCURACY FOR UPOS TAGS ON TEST DATA ######################")
    print(accuracy_word_test)
    accuracy_sentence_test = accuracy_sentence_level(true_label_test, predicted_values_test, token_list_test, Number_of_sentences_test)
    print("############## SENTENCE_LEVEL ACCURACY FOR UPOS TAGS ON TEST DATA ######################")
    print(accuracy_sentence_test)
    ###########################################################################################################
    #%%
    unique_token_list_XPOS = list(Counter_XPOS.keys())
    store_probabilities_XPOS = build_store_probabilities(token_with_OOV, XPOS_list)
    label_tags_XPOS = give_tag_labels(unique_token_list_XPOS)
    print(label_tags_XPOS)
    dictionary_count_XPOS, transition_matrix_XPOS = transition_probability_matrix(XPOS_list, Counter_XPOS, unique_token_list_XPOS, label_tags_XPOS)
    emission_matrix_XPOS = get_emission_matrix(store_probabilities_XPOS, token_with_OOV, label_tags_XPOS, Counter_XPOS)
    initial_probabiltiy_XPOS = []
    #%%
    for  i in range(len(label_tags_XPOS)):
        initial_probabiltiy_XPOS.append(1/(len(label_tags_XPOS)))   #uniform distribution
    initial_probabiltiy_XPOS = np.array(initial_probabiltiy_XPOS)
    print(initial_probabiltiy_XPOS)
    
    #%%
    predicted_values_dev_XPOS = []
    for u in range(len(sentences_dev)):
        observations_1_XPOS = sentences_dev[u]
        final_sequence_XPOS, score_XPOS = viterbi(initial_probabiltiy_XPOS, transition_matrix_XPOS, emission_matrix_XPOS, observations_1_XPOS, token_with_OOV, token_types)
        predicted_values_dev_XPOS.append(final_sequence_XPOS)
        
    predicted_values_dev_XPOS = np.array(predicted_values_dev_XPOS)
    #print(predicted_values_dev_XPOS.shape)
    predicted_values_dev_XPOS = np.hstack(predicted_values_dev_XPOS)
    true_label_dev_XPOS = give_true_label_list(label_tags_XPOS, XPOS_list_dev)
    accuracy_word_dev_XPOS = accuracy_word_level(true_label_dev_XPOS, predicted_values_dev_XPOS)
    print("############## WORD LEVEL ACCURACY FOR XPOS TAGS ON DEV DATA ######################")
    print(accuracy_word_dev_XPOS)
    accuracy_sentence_dev_XPOS = accuracy_sentence_level(true_label_dev_XPOS, predicted_values_dev_XPOS, token_list_dev, Number_of_sentences_dev)
    print("############## SENTENCE_LEVEL ACCURACY FOR XPOS TAGS ON DEV DATA ######################")
    print(accuracy_sentence_dev_XPOS)
    
    predicted_values_test_XPOS = []
    for u1 in range(len(sentences_test)):
        observations_1_test_XPOS = sentences_test[u1]
        final_sequence_test_XPOS, score = viterbi(initial_probabiltiy_XPOS, transition_matrix_XPOS, emission_matrix_XPOS, observations_1_test_XPOS, token_with_OOV, token_types)
        predicted_values_test_XPOS.append(final_sequence_test_XPOS)
        
    predicted_values_test_XPOS = np.array(predicted_values_test_XPOS)
    predicted_values_test_XPOS = np.hstack(predicted_values_test_XPOS)
    true_label_test_XPOS = give_true_label_list(label_tags_XPOS,XPOS_list_test)
    accuracy_word_test_XPOS = accuracy_word_level(true_label_test_XPOS, predicted_values_test_XPOS)
    print("############## WORD LEVEL ACCURACY FOR XPOS TAGS ON TEST DATA ######################")
    print(accuracy_word_test_XPOS)
    accuracy_sentence_test_XPOS = accuracy_sentence_level(true_label_test_XPOS, predicted_values_test_XPOS, token_list_test, Number_of_sentences_test)
    print("############## SENTENCE_LEVEL ACCURACY FOR XPOS TAGS ON TEST DATA ######################")
    print(accuracy_sentence_test_XPOS)


if __name__ == '__main__':
    main()
    #
#Number_of_sentences, token_list , UPOS_list, XPOS_list = reading_data_token_types(data_read_train)
#print(Number_of_sentences)
#corpora_word_count = Counter(token_list)
#token_with_OOV = introduce_OOV(token_list, THRESHOLD, corpora_word_count)
#new_corpora_word_count = Counter(token_with_OOV)
#token_types = list(new_corpora_word_count.keys())
#V = len(new_corpora_word_count)
#Counter_UPOS= dict(Counter(UPOS_list))
#Counter_XPOS = dict(Counter(XPOS_list))
#Prior_UPOS = prior_probability(Counter_UPOS)
#Prior_XPOS = prior_probability(Counter_XPOS)
#
#
#
#
#unique_token_list = list(Counter_UPOS.keys())
#label_tags = give_tag_labels(unique_token_list)
#store_probabilities = build_store_probabilities(token_with_OOV, UPOS_list)
#dictionary_count, transition_matrix = transition_probability_matrix(UPOS_list, Counter_UPOS, unique_token_list, label_tags)
#print(label_tags)
#emission_matrix = get_emission_matrix(store_probabilities, token_with_OOV, label_tags, Counter_UPOS)
#initial_probabiltiy = []
#for  i in range(len(label_tags)):
#    initial_probabiltiy.append(1/(len(label_tags)))   #uniform distribution
#initial_probabiltiy = np.array(initial_probabiltiy)
#print(initial_probabiltiy)
## print(emission_matrix[0])
##%%
#data_dev = open(DATA_LOCATION+"\\el_gdt-ud-dev.conllu", 'r', encoding="utf-8")
#data_dev_read = data_dev.readlines()
#Number_of_sentences_dev, token_list_dev, UPOS_list_dev, XPOS_list_dev = reading_data_token_types(data_dev_read)
#sentences_dev = []
#temp_list = []
#for i in range(len(token_list_dev)):
#    if token_list_dev[i] == "EOS":
#        temp_list.append("EOS")
#        sentences_dev.append(temp_list)
#        temp_list = []
#    else:
#        temp_list.append(token_list_dev[i])
#print(len(sentences_dev))
#print(Number_of_sentences_dev)
#sentences_dev = np.array(sentences_dev)
#predicted_values_dev = []
#for u in range(len(sentences_dev)):
#    observations_1 = sentences_dev[u]
#    final_sequence, score = viterbi(initial_probabiltiy, transition_matrix, emission_matrix, observations_1, token_with_OOV, token_types)
##    print(final_sequence)
#    predicted_values_dev.append(final_sequence)
#    
#predicted_values_dev = np.array(predicted_values_dev)
#print(predicted_values_dev.shape)
#predicted_values_dev = np.hstack(predicted_values_dev)
#true_label = give_true_label_list(label_tags,UPOS_list_dev)
#accuracy_word_dev = accuracy_word_level(true_label, predicted_values_dev)
#print(accuracy_word_dev)
#accuracy_sentence_dev = accuracy_sentence_level(true_label, predicted_values_dev, token_list_dev, Number_of_sentences_dev)
#print(accuracy_sentence_dev)
#
##%%
#data_test = open(DATA_LOCATION+"\\el_gdt-ud-test.conllu", 'r', encoding="utf-8")
#data_test_read = data_test.readlines()
#Number_of_sentences_test, token_list_test, UPOS_list_test, XPOS_list_test = reading_data_token_types(data_test_read)
#sentences_test = []
#temp_list_test = []
##%%
#for i in range(len(token_list_test)):
#    if token_list_test[i] == "EOS":
#        temp_list_test.append("EOS")
#        sentences_test.append(temp_list_test)
#        temp_list_test = []
#    else:
#        temp_list_test.append(token_list_test[i])
#print(len(sentences_test))
#print(Number_of_sentences_test)
#sentences_test = np.array(sentences_test)
##%%
#predicted_values_test = []
#for u1 in range(len(sentences_test)):
#    observations_1_test = sentences_test[u1]
#    final_sequence_test, score = viterbi(initial_probabiltiy, transition_matrix, emission_matrix, observations_1_test, token_with_OOV, token_types)
##    print(final_sequence_test)
#    predicted_values_test.append(final_sequence_test)
#    
#predicted_values_test = np.array(predicted_values_test)
#print(predicted_values_test.shape)
#predicted_values_test = np.hstack(predicted_values_test)
#true_label_test = give_true_label_list(label_tags,UPOS_list_test)
#accuracy_word_test = accuracy_word_level(true_label_test, predicted_values_test)
#print(accuracy_word_test)
#accuracy_sentence_test = accuracy_sentence_level(true_label_test, predicted_values_test, token_list_test, Number_of_sentences_test)
#print(accuracy_sentence_test)
############################################################################################################
##%%
#unique_token_list_XPOS = list(Counter_XPOS.keys())
#store_probabilities_XPOS = build_store_probabilities(token_with_OOV, XPOS_list)
#label_tags_XPOS = give_tag_labels(unique_token_list_XPOS)
#print(label_tags_XPOS)
#dictionary_count_XPOS, transition_matrix_XPOS = transition_probability_matrix(XPOS_list, Counter_XPOS, unique_token_list_XPOS, label_tags_XPOS)
#emission_matrix_XPOS = get_emission_matrix(store_probabilities_XPOS, token_with_OOV, label_tags_XPOS, Counter_XPOS)
#initial_probabiltiy_XPOS = []
##%%
#for  i in range(len(label_tags_XPOS)):
#    initial_probabiltiy_XPOS.append(1/(len(label_tags_XPOS)))   #uniform distribution
#initial_probabiltiy_XPOS = np.array(initial_probabiltiy_XPOS)
#print(initial_probabiltiy_XPOS)
#
##%%
#predicted_values_dev_XPOS = []
#for u in range(len(sentences_dev)):
#    observations_1_XPOS = sentences_dev[u]
#    final_sequence_XPOS, score_XPOS = viterbi(initial_probabiltiy_XPOS, transition_matrix_XPOS, emission_matrix_XPOS, observations_1_XPOS, token_with_OOV, token_types)
#    predicted_values_dev_XPOS.append(final_sequence_XPOS)
#    
#predicted_values_dev_XPOS = np.array(predicted_values_dev_XPOS)
##print(predicted_values_dev_XPOS.shape)
#predicted_values_dev_XPOS = np.hstack(predicted_values_dev_XPOS)
#true_label_dev_XPOS = give_true_label_list(label_tags_XPOS, XPOS_list_dev)
#accuracy_word_dev_XPOS = accuracy_word_level(true_label_dev_XPOS, predicted_values_dev_XPOS)
#print(accuracy_word_dev_XPOS)
#accuracy_sentence_dev_XPOS = accuracy_sentence_level(true_label_dev_XPOS, predicted_values_dev_XPOS, token_list_dev, Number_of_sentences_dev)
#print(accuracy_sentence_dev_XPOS)
#
##%%
#predicted_values_test_XPOS = []
#for u1 in range(len(sentences_test)):
#    observations_1_test_XPOS = sentences_test[u1]
#    final_sequence_test_XPOS, score = viterbi(initial_probabiltiy_XPOS, transition_matrix_XPOS, emission_matrix_XPOS, observations_1_XPOS, token_with_OOV, token_types)
##    print(final_sequence_test)
#    predicted_values_test_XPOS.append(final_sequence_test_XPOS)
#    
#predicted_values_test_XPOS = np.array(predicted_values_test_XPOS)
#print(predicted_values_test_XPOS.shape)
#predicted_values_test_XPOS = np.hstack(predicted_values_test_XPOS)
#true_label_test_XPOS = give_true_label_list(label_tags_XPOS,XPOS_list_test)
#accuracy_word_test_XPOS = accuracy_word_level(true_label_test_XPOS, predicted_values_test_XPOS)
#print(accuracy_word_test_XPOS)
#accuracy_sentence_test_XPOS = accuracy_sentence_level(true_label_test_XPOS, predicted_values_test_XPOS, token_list_test, Number_of_sentences_test)
#print(accuracy_sentence_test_XPOS)
#
