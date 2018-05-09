import os
import cv2
import numpy as np
# find the most likely character by template matching
def recognize_character(char):
    ch_height, ch_width = char.shape
    # add zero padding to left and right side, to allow for
    # correlation sliding
    padding = np.zeros((char.shape[0], char.shape[1]))
    char = np.concatenate((padding,char), axis = 1)
    char = np.concatenate((char,padding), axis = 1)
    char = char.astype(np.uint8)
    correlations = []
    templates = os.listdir('templates')
    kernel = np.ones((5, 5), np.uint8)

    for i in range(len(templates)):
        # load template
        path = 'templates/' + templates[i]
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # binarize the template
        _, template = cv2.threshold(template,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # resize the template to the character size
        template = cv2.resize(template, (ch_width, ch_height))
        # compute the normalized cross correlation
        res = cv2.matchTemplate(char,template,cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        correlations.append(max_val)


    # get the 3 characters with the highest correlation
    best1 = correlations.index(max(correlations))
    correlations[best1] = 0
    best2 = correlations.index(max(correlations))
    correlations[best2] = 0
    best3 = correlations.index(max(correlations))
    print(best1, best2, best3)
    
    return [best1, best2, best3]
