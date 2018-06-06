import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from recognize_character_cnn import *
import itertools
import pandas as pd


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
    #print(gram, 'weighted frequency:',  count)
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



# recognize the characters in a word by sliding a window across big
# connected components and recognizing characters within
def recognize_word(word, avg_width, trans_matrix, start_vector, plot):
    
    # show the word
    if(plot):
        plt.figure(figsize = (500,4))
        plt.imshow(word, cmap='gray', aspect = 1)
        plt.show()
    

    # define how much the window moves each step
    stride = 2
    
    # define the window size and location
    window_height = 50
    window_width = 50
    window_x = 0
    window_y = 0
    
    word = 255 - word
    

    kernel = np.ones((3,3),np.uint8)
   
    # find the connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    # remove background component
    stats = stats[1:]
    
    stats = stats[stats[:,0].argsort()]
    components = [] 
    # plot all components
    for stat in stats:
        
        if(stat[2] > (0.4 * avg_width) and stat[3] > (0.2 * avg_width) and stat[2] < (avg_width * 5) and stat[3] < (avg_width * 5)):
            print(stat[0])
            components.append(stat)
          
    candidates = [] 
    n_characters = 0
    components = list(np.flip(np.array(components), 0))
    # for every component:
    for component in components:
       
        # slide a window across and classify the content within
        component_image = word[component[1]: component[1]+ component[3], component[0]: component[0]+component[2]]
        word_width = component_image.shape[1]
        # dilate the component
        #component_image = cv2.dilate(component_image, kernel, iterations = 1)
        # if the component is big enough, slide through it to detect multiple characters
        if(component[2] > (avg_width * 1.4)):
            print("Component is big: use sliding window!")
            # add zero padding to the left and right of image
            padding = np.zeros((component_image.shape[0], int(component_image.shape[1]/6)))
            component_image = np.concatenate((padding,component_image), axis = 1)
            component_image = np.concatenate((component_image,padding), axis = 1)
            component_image = component_image.astype(np.uint8)
            
            window_width = int(component[2] / 2)
            
            # show the augmented component image
            if(plot):
                plt.imshow(component_image, cmap='gray', aspect = 1)
                plt.show()

            # slide the window across the image, until the end of the image
            # has been reached
            
            # show initial window content
            window = component_image[window_y:window_y + component_image.shape[0], window_x:window_x + window_width]
            print("Initial window:")
            if(plot):
                plt.figure(figsize = (500,4))
                plt.imshow(window, cmap='gray', aspect = 1)
                plt.show()
            
            # keep track of how many candidates there are per window classification
            N_candidates = 0
            previous_N = 100
            possible_character = False
            
            window_chars = []
            while((window_x + window_width) <= component_image.shape[1]):
                window = component_image[window_y:window_y + component_image.shape[0], window_x:window_x + window_width]
                # skip complete empty windows
                if(np.count_nonzero(window) != 0):
                   
                    # try to recognize characters within the component
                    top_chars, top_scores = recognize_character(window)
                    N_candidates = len(top_chars)
                    print("Window position", int(window_x / stride) ,"recognition:", top_chars, top_scores)
                    # If character was possibly found in previous window position,
                    # and the amount of candidates is going up again,
                    #take the previous window as the character
                    if((possible_character == True and N_candidates > previous_N) or top_scores[0] >= 0.7):
                        
                        # go back one stride, unless the window is at the first position
                        if(window_x != 0):
                            print("\nCharacter found at position", int((window_x - stride) / stride), ":")
                            window = component_image[window_y:window_y + component_image.shape[0], window_x - stride :window_x -stride + window_width + 15]
                            window = cv2.dilate(window ,kernel, iterations = 1)
                        else:
                            print("\nCharacter found at position", 0, ":")
                        
                        
                        #show the window content
                        if(plot):
                            plt.figure(figsize = (500,4))
                            plt.imshow(window, cmap='gray', aspect = 1)
                            plt.show()
                        
                        top_chars, top_scores = recognize_character(window)
                        
                        # store window-slided detected characters: these need to be stored in the reversed order later
                        window_chars.append(top_chars)
                        #candidates.append(top_chars)
                        
                        print("Recognition result:", top_chars, top_scores)
                        print("---------------------------------------------------\n")
                        possible_character = False
                        
                        # skip a few frames ahead, because next character
                        # won't be only a few pixels next to current character
                        #window_x += int(avg_width) - stride
                        window_x += int(word_width / 2) - stride + 15
                        
                    
                    # if the amount of candidates is smaller, 
                    # character might be better detected
                    if(N_candidates <= previous_N):
                        possible_character = True
                    else:
                        possible_character = False
                        
                    
                    
                    previous_N = N_candidates

                window_x += stride
                
            window_chars.reverse()
            for character in window_chars:
                candidates.append(character)
        # else, sliding is not necessary as the component is probably just a character,
        # which can be recognized immediately
        else:
            print("Component small enough,  proceed to recognition:")
            # put a window around the component
            #width = 0
            #height = 0
            """if(component[2] > window_width):
                width = component[2]
            else:
                width = window_width"""
                
            if(component[3] > window_height):
                height = component[3]
            else:
                height = window_height
            
            
           
            window = word[component[1]:component[1] + height, component[0]:component[0] + component[2]]
            window = cv2.dilate(window ,kernel, iterations = 1)
            if(plot):
                plt.imshow(window, cmap='gray', aspect = 1)
                plt.show()
            top_chars, top_scores = recognize_character(window)
            candidates.append(top_chars)
            print("Recognition result:", top_chars, top_scores)
            print("---------------------------------------------------\n")
        print()
    print("Candidates:", candidates)
    
    combinations = list(itertools.product(*candidates))
    print("Combinations:", combinations)
    
    # find the most frequent combination
    ngrams = pd.read_excel('n_grams_probs.xlsx')
    names = list(ngrams['Names'])
    freqs = list(ngrams['Frequencies'])
    names = list(map(lambda x: x.lower(), names))
    
    best_prob_index = 0
    best_prob = 0
    
    combi_props = []
    
    
    
    
    for combi in combinations:    
        #freq = compute_frequency("_".join(combi), len(candidates), names, freqs)
        
        # first, get the probability of a word starting with the character\     
        try:
            prob = start_vector[combi[-1]]
        except:
            prob = 0
            
        # multiply the starting probability with the transition probabilities
        for i in range(len(combi) - 1):
            current_char = combi[i]
            next_character = combi[i + 1]
            try:
                prob *= trans_matrix[current_char][next_character]
            except:
                prob = 0
        
                
   
        if(prob > best_prob):
            #best_freq = freq
            #best_freq_index = combinations.index(combi)        
            best_prob = prob
            best_prob_index = combinations.index(combi)
        print("Combi:", combi, "Probability:", prob)
    
    best = "_".join(list(combinations[best_prob_index]))
    print("Best word:", best)
    
    if(best in names):
        index = names.index(best)  
        print(list(ngrams['Hebrew_character'])[index])
        return list(ngrams['Hebrew_character'])[index]
        #return best
    else:
        # if word is not in ngrams, write all of its characters to file
        best = best.split('_')
        #if(len(best) > 1):
            #best.reverse()
        print("Best?!:", best)
        return best
        
  
        
        
        

    
    
    