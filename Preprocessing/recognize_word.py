import os
import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from recognize_character_cnn import *


# calculate the posterior probability of each character - P(ngram/char)
def calculate_posterior(names, freqs):
    ngrams = pd.read_excel('ngrams_frequencies_withNames.xlsx')
    names = list(ngrams['Names'])
    freqs = list(ngrams['Frequencies'])
    names = list(map(lambda x: x.lower(), names))

    char_list = []
    filtered_names = []
    filtered_freqs = []
    for i in range(len(names)):
        freq = freqs[i]
        if freq >= 10:
            word = names[i]
            filtered_names.append(word)
            filtered_freqs.append(freq)

            char_in_names = word.split('_')
            char_list = char_list + char_in_names

    unique_char_list, counts = np.unique(np.array(char_list), return_counts=True)
    counts *= 0
    dict_char = dict(zip(unique_char_list, counts))

    for i in range(len(filtered_names)):
        freq = filtered_freqs[i]
        word_c_list = filtered_names[i].split('_')
        chars, counts = np.unique(np.array(word_c_list), return_counts=True)
        counts = counts * freq
        for j in range(len(chars)):
            dict_char[chars[j]] += counts[j]

    sum_all_char_freq = sum(dict_char.values())
    sum_all_freqs = np.array(filtered_freqs).sum()

# probability of each char
    for char in unique_char_list:
        dict_char[char] = dict_char[char] / sum_all_char_freq

    posterior_matrix = np.zeros([len(unique_char_list), len(filtered_names)])
    for i in range(len(filtered_names)):
        word = filtered_names[i]
        freq = filtered_freqs[i]
        pro_word = freq / sum_all_freqs
        chars_in_word = word.split('_')
        pro_char_in_word = 1 / len(chars_in_word)

        for j in range(len(unique_char_list)):
            char = unique_char_list[j]
            if char in chars_in_word:
                posterior_matrix[j][i] = pro_word * pro_char_in_word / dict_char[char]

    print(posterior_matrix)
    return posterior_matrix, unique_char_list, filtered_names

# for each ngram, check if they occur as ngrams for words in the ngram file
def find_Ngram(gram, current_n, names, frequencies):
    count = 0
    for word in names:
        # take the percentage of the ngram out of the entire word
        proportion = current_n / len(word.split('_'))
        # compare gram with all current_n grams of each word
        if gram in word:
            index = names.index(word)
            count += proportion * frequencies[index]
    # print(gram, 'weighted frequency:',  count)
    return count


# compute the frequency of the word by summing the frequency of all of its ngrams
def compute_frequency(word_string, n_chars, names, frequencies):
    # a word with N characters has N-1 bigrams
    N_grams = n_chars - 1
    current_n = 2
    total_sum = 0
    while current_n <= n_chars:
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
        if combination[i] == char_proposals[i][1]:
            penalty -= 0.2
        # penalize more if character is a third choice
        elif combination[i] == char_proposals[i][2]:
            penalty -= 0.3
    return penalty    


# find the most likely word from the given word image
def recognize_word(word, avg_width):
    kernel = np.ones((3, 3), np.uint8)
    
    plt.figure(figsize = (500,4))
    plt.imshow(word, cmap='gray', aspect = 1)
    plt.show()
    
    templates = os.listdir('templates')
    ngrams = pd.read_excel('ngrams_frequencies_withNames.xlsx')
    # print(ngrams.head())
    # _, word = cv2.threshold(word,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    word = 255 - word
    word = cv2.dilate(word, kernel, iterations = 1)
    
    # find the characters by using connected components 
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    # remove the background component
    stats = stats[1:]
    # order the components based on x position
    stats = stats[stats[:,0].argsort()]
    chars = []
    # filter out components that are too small
    print(stats)
    print("Average width:", avg_width)
    for i in range(len(stats)):
        cutoff_points = []
        if stats[i][2] > (0.4 * avg_width and stats[i][2] < (avg_width * 5)):
            width = stats[i][2]
            height = stats[i][3]
            # If component is too wide, it probably consists of multiple characters
            # apply water drop algorithm to try to separate all the characters within 
            if width > (1.3 * avg_width):
                # sub_word = word[stats[i][1]:stats[i][1] + height, stats[i][0]:stats[i][0] + width]
                sub_word = word[:, stats[i][0]:stats[i][0] + width]
                histogram = np.zeros(width)
                # count black pixels for every column of the characer(s)
                for i in range(width):
                    histogram[i] = height - np.count_nonzero(sub_word[:, i])
                mx = max(histogram)
                mn = min(histogram)
                avg = np.mean(histogram)
                std = np.std(histogram)

                # if a column has too few black pixels, separate the two components
                for i in range(len(histogram)):
                    if histogram[i] <= (avg - (1.5 * std) and histogram[i] > (0.2*height) and len(cutoff_points) < 3):
                        
                        cutoff_points.append(i)
                # if separation is needed, separate the component using all the found
                # cutoff points
                x1 = 0
                x2 = 0
                for point in cutoff_points:
                    x2 = point
                    # character has to be big enough
                    if (x2 - x1) > (0.3 * avg_width):
                        chars.append(sub_word[:, x1:x2])
                        x1 = point  
                    chars.append(sub_word[:, x1:])
                    
                # if no cutoff points found, append the 'character' anyway
                if len(cutoff_points) == 0:
                    chars.append(sub_word)

            else:
                # chars.append(word[stats[i][1]:stats[i][1] + height, stats[i][0]:stats[i][0] + width])
                chars.append(word[:, stats[i][0]:stats[i][0] + width])
    word_string = ""
    # match the characters in the word
    char_proposals = [] 
    for char in chars:
        #char = word[stat[1]:stat[1] + stat[3], stat[0]:stat[0]+stat[2]]
        predictions, scores = recognize_character(char)
        char_proposals.append(predictions)
    # print(tuple(char_proposals))
    # print(char_proposals)
    # find every possible combination of possible characters
    combinations = list(itertools.product(*char_proposals))
    # print(len(stats), "words, ", str(len(combinations)), " possible combinations")

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
            print(char_index)
            ch = templates[char_index]
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
        combi_freqs.append([freq, word_string])
        # print('Total:', freq)

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

        
calculate_posterior(None, None)

