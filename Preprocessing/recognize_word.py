import os
import cv2
import numpy as np
import pandas as pd
from recognize_character import *

# find the most likely word from the given word image
def recognize_word(word):
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
    for i in range(len(chars)):
        char = word[stats[i][1]:stats[i][1]+stats[i][3],
                    stats[i][0]:stats[i][0]+stats[i][2]]
        corr, ch = recognize_character(char)
        word_string = word_string + ch + " "
    # reverse the string
    word_string = word_string.split()
    word_string.reverse()
    word_string = "_".join(word_string)
    print(word_string)
    #print(ngrams['Names'])
    names = list(ngrams['Names'])
    freqs = list(ngrams['Frequencies'])
    try:
        # find the index of the recognized word in the ngrams file
        index = names.index(word_string)
        # find the frequency of the word
        freq = freqs[index]
        #print(word_string, "has a frequency of", freq)
    except:
        print("Word does not exist in ngrams file")
   
        
        
   
    
