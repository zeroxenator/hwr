import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from recognize_character_cnn import *

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
    

    kernel = np.ones((5,5),np.uint8)
   
    # find the connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    # remove background component
    stats = stats[1:]
    components = [] 
    # plot all components
    for stat in stats:
        if(stat[2] > (0.5 * avg_width) and stat[3] > (0.5 * avg_width) and stat[2] < (avg_width * 5) and stat[3] < (avg_width * 5)):
            components.append(stat)
    
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
                            print("\nCharacter found at position", 0, ":")
                            window = component_image[window_y:window_y + component_image.shape[0], window_x - stride :window_x -stride + window_width]
                        print("\nCharacter found at position", int((window_x - stride) / stride), ":")
                        
                        #show the window content
                        plt.figure(figsize = (500,4))
                        plt.imshow(window, cmap='gray', aspect = 1)
                        plt.show()
                        
                        top_chars, top_scores = recognize_character(window)
                        
                        print("Recognition result:", top_chars, top_scores)
                        print("---------------------------------------------------\n")
                        possible_character = False
                        
                        # skip a few frames ahead, because next character
                        # won't be only a few pixels next to current character
                        #window_x += int(avg_width) - stride
                        window_x += int(word_width / 2) - stride
                        
                    
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
            plt.imshow(component_image, cmap='gray', aspect = 1)
            plt.show()
            top_chars, top_scores = recognize_character(component_image)
            print("Recognition result:", top_chars, top_scores)
            print("---------------------------------------------------\n")
        print()

    
    
    