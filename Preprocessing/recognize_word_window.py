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
def recognize_word(word, avg_width):
    
    # show the word
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
        
        if(stat[2] > (0.5 * avg_width) and stat[3] > (0.5 * avg_width) and stat[2] < (avg_width * 5) and stat[3] < (avg_width * 5)):
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
            plt.imshow(component_image, cmap='gray', aspect = 1)
            plt.show()

            # slide the window across the image, until the end of the image
            # has been reached
            
            # show initial window content
            window = component_image[window_y:window_y + component_image.shape[0], window_x:window_x + window_width]
            print("Initial window:")
            plt.figure(figsize = (500,4))
            plt.imshow(window, cmap='gray', aspect = 1)
            plt.show()
            
            # keep track of how many candidates there are per window classification
            N_candidates = 0
            previous_N = 100
            possible_character = False
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
                        plt.figure(figsize = (500,4))
                        plt.imshow(window, cmap='gray', aspect = 1)
                        plt.show()
                        
                        top_chars, top_scores = recognize_character(window)
                        candidates.append(top_chars)
                        
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
        # else, sliding is not necessary as the component is probably just a character,
        # which can be recognized immediately
        else:
            print("Component small enough,  proceed to recognition:")
            component_image = cv2.dilate(component_image ,kernel, iterations = 1)
            plt.imshow(component_image, cmap='gray', aspect = 1)
            plt.show()
            top_chars, top_scores = recognize_character(component_image)
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
    
    best_freq_index = 0
    best_freq = 0
    
    for combi in combinations:    
        freq = compute_frequency("_".join(combi), len(candidates), names, freqs)
        if(freq > best_freq):
            best_freq = freq
            best_freq_index = combinations.index(combi)
        print("Combi:", combi, "Frequency:", freq)
    
    best = "_".join(list(combinations[best_freq_index]))
    print("Best word:", best)
    
    if(best in names):
        print("Best is in names!")
        index = names.index(best)  
        print(list(ngrams['Hebrew_character'])[index])
        return list(ngrams['Hebrew_character'])[index]
    else:
        return best
        
  
        
        
        

    
    
    