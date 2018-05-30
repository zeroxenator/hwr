import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from CNN_test.classify import *

# find the most likely character by template matching
def recognize_character(char):
    # print("Recognizing the following character:")
    #kernel = np.ones((3,3),np.uint8)
    #char = cv2.erode(char, kernel, iterations = 1)
    user_defined_threshold = 0.08

    top_chars, top_scores = recognition_char(char, user_defined_threshold)

    # templates = [string.split('.')[0].lower() for string in os.listdir('templates')]
    # dict_chars = dict(zip(templates, range(len(templates))))
    # top_chars = [dict_chars[char.split(' ')[0]] for char in top_chars]
    # print(top_chars)

    return top_chars, top_scores

# image_path = 'C:/Users/Andrew X/Documents/GitHub/handwriting_recognition/Preprocessing/CNN_test/test2.jpg'
# # image_path = 'C:/Users/Andrew X/Documents/GitHub/handwriting_recognition/Preprocessing/image-output/test18_binarized.jpg'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# # recognize_handwriting(image)
# recognize_character(image)
