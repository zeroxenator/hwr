import os
import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_predictions(model, image, n_chars):
     # use the lstm for prediction
    output = model.predict(image)
    #print("Output:",output)
    output_sorted = np.sort(output)
    #output_sorted = output_sorted[0]
    #print("sorted output:", output_sorted)
    
    output_trimmed = output_sorted[:, 27 - n_chars:]
    
    print("Trimmed output:", output_trimmed)
    
    mn = np.min(output_trimmed)
    #mn = 0.4
    output_trimmed = np.where(output >= mn)
    output_trimmed = output_trimmed[1]
    
    #print("Best indices:", output_trimmed)
    final_output = []
    
    labels = {0: "alef", 1: "ayin",  2: "bet",  3: "dalet",  4: "gimel",  5: "he",  6: "het", 
                     7: "kaf",  8: "kaf-final",  9: "lamed",  10: "mem",  11: "mem-medial",  12: "nun-final", 
                     13: "nun-medial",  14: "pe",  15: "pe-final",  16: "qof",  17: "resh",  18: "samekh",  19: "shin", 
                     20: "taw",  21: "tet",  22: "tsadi-final",  23: "tsadi-medial",  24: "waw",  25: "yod",  26: "zayin"
    }
    
  
    
    
    print("Idx:", output_trimmed)
    for idx in output_trimmed:
        final_output.append(labels[idx])
    #print("Output:", output_trimmed)  
    return final_output
    
    



# recognize the word using the trained LSTM model 
def recognize_word_lstm(word, model, avg_width, plot):

    
    
    kernel = np.ones((3,3))
    
    # invert the colors
    word = 255 - word
    print("orig shape:",word.shape)
    
    ret,word = cv2.threshold(word,127,255,cv2.THRESH_BINARY)
    
    #word = cv2.dilate(word, kernel, iterations = 1)
    #word = cv2.erode(word, kernel, iterations = 1)
    
    
    # show the word
    if(plot):
        """im = cv2.imread('4.jpg', 0)
        
        plt.figure(figsize = (500,4))
        plt.imshow(im, cmap='gray', aspect = 1)
        plt.show()"""
        
        plt.figure(figsize = (500,4))
        plt.imshow(word, cmap='gray', aspect = 1)
        plt.show()
    
    
    # try to count characters in word
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    stats = stats[1:]
    
      
    characters = []
    
    n_chars = 0
    # save and count characters
    for component in stats:     
        print(component)
        if(component[4] > 300):
            n_chars += 1
            character = word[component[1]:component[1] + component[3],component[0]:component[0] + component[2]]
            # plot the characters
            plt.figure(figsize = (500,4))
            plt.imshow(character, cmap='gray', aspect = 1)
            plt.show()
                
            character = cv2.resize(character, (32, 32))
            character = character / 255
            character = np.reshape(character, (1,32,32))
            characters.append(character)
    
    print("N chars:", n_chars)
  
    word = cv2.resize(word, (32, 32))
    word = word / 255
    word = np.reshape(word, (1,32,32))
    
    print("Word-wise predictions:", get_predictions(model, word, 4))
    for ch in characters:
        print("Character recognition:", get_predictions(model, ch, 1))
    
    
    
    
