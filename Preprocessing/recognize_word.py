import os
import cv2
import itertools
import numpy as np
import pandas as pd
from recognize_character import *

# for each ngram, check if they occur as ngrams for words in the ngram file
def find_Ngram(gram, current_n, names, frequencies):
    count = 0
    for word in names:
        #take the percentage of the ngram out of the entire word
        proportion = current_n / len(word.split('_'))
        # compare gram with all current_n grams of each word
        if(gram in word):
            index = names.index(word)
            count += proportion * frequencies[index]
    print(gram, 'weighted frequency:',  count)
    return count
            
        
        
    


# compute the frequency of the word by summing the frequency of all of its ngrams
def compute_frequency(word_string, n_chars, names, frequencies):
    # a word with N characters has N-1 bigrams
 
    N_grams = n_chars - 1
    current_n = 2
    total_sum = 0
    while(current_n <= n_chars):
        start = 0
        end = current_n
        # for each ngram, check if they occur as ngrams for words in the ngram file
        for i in range(N_grams):                  
            gram = "_".join(word_string[start:end])
            gram = gram.lower()
            total_sum += find_Ngram(gram, current_n, names, frequencies)  
            """try:
                 # find the index of the recognized word in the ngrams file
                index = names.index(gram)
                # find the frequency of the word
                freq = freqs[index]
                print("N gram:", N_grams, "index:", i)
                total_sum += freq
            except:
                pass
            """
            start += 1
            end += 1
          
        current_n += 1
        N_grams -= 1
        
    return total_sum
    
# compute a penalty based on whether the characters in the combination were first, second or third choice
def compute_penalty(combination, char_proposals):
    penalty = 1
    for i in range(len(combination)):
        # no penalty if character is a first choice
        # penalize a bit if character is a second choice
        if(combination[i] == char_proposals[i][1]):
            penalty -= 0.2
        # penalize more if character is a third choice
        elif(combination[i] == char_proposals[i][2]):
            penalty -= 0.3
    return penalty    
        
        
    

# find the most likely word from the given word image
def recognize_word(word):
    templates = os.listdir('templates')
    ngrams = pd.read_excel('ngrams_frequencies_withNames.xlsx')
    #print(ngrams.head())
    _, word = cv2.threshold(word,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    word = 255 - word
    # find the characters by using connected components 
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    # remove the background component
    stats = stats[1:]
    # order the components based on x position
    stats = stats[stats[:,0].argsort()]
    chars = []
    # filter out components that are too small
    for i in range(len(stats)):
        if(stats[i][4] > 300):
            chars.append(stats[i])
    word_string = ""
    # match the characters in the word
    char_proposals = [] 
    for i in range(len(chars)):
        char = word[stats[i][1]:stats[i][1]+stats[i][3],
                    stats[i][0]:stats[i][0]+stats[i][2]]
        predictions = recognize_character(char)
        char_proposals.append(predictions)
    #print(tuple(char_proposals))
    print(char_proposals)
    # find every possible combination of possible characters
    combinations = list(itertools.product(*char_proposals))
    print(len(stats), "words, ", str(len(combinations)), " possible combinations")

    # find the most frequent combination
    combi_freqs = []
    names = list(ngrams['Names'])
    freqs = list(ngrams['Frequencies'])
    
    names = list(map(lambda x: x.lower(), names))
    
    for combination in combinations:
        word_string = ""
        n_chars = 0
        for char_index in combination:
            n_chars += 1
            # get the character name
            ch = templates[char_index].split('.')[0]
            # add the character to the total word string
            word_string = word_string + ch + " "
        # reverse the string
        word_string = word_string.split()
        word_string.reverse()
        word_string = "_".join(word_string)
        
        # compute a penalty for a combination, depending on how many characters from
        # 'second' or 'third' most likely are contained within the combination
        penalty = compute_penalty(combination, char_proposals)
        
        
        # compute the total frequency of the word
        freq = compute_frequency(word_string.split('_'), n_chars, names, freqs)
        freq = penalty * freq
        combi_freqs.append([freq,word_string])
        print('Total:', freq)

    # return the most frequent combination
    mx = 0
    best_combi = []
    # go in backwards-order, such that when none of the combinations
    # appear in the ngram file, then simply the combination
    # of most correlated characters will be returned
    i = len(combi_freqs) - 1
    while(i >= 0):
        if(combi_freqs[i][0] >= mx):
            mx = combi_freqs[i][0]
            best_combi = combi_freqs[i][1]
        i -= 1
    return best_combi

   
        
        
      
        
   
    
