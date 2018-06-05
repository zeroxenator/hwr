import pandas as pd
from collections import defaultdict

ngrams = pd.read_excel('n_grams_probs.xlsx')
names = list(ngrams['Names'])
names = list(map(lambda x: x.lower(), names))
freqs = list(ngrams['Frequencies'])

memory = {}
freq_memory = {}


def learn_key(key, value, freq):
    # use first chars in bigrams as keys and all second chars as its values
    if key not in memory:
        memory[key] = []
    memory[key].append(value)
    
    # use first chars in bigrams as keys and the freqs in exel as its values
    if key not in freq_memory:
        freq_memory[key] = []
    freq_memory[key].append(freq)


def store_bigrams():
    for i in range(len(names)):
        word = names[i]
        char_in_word = word.split('_')

        # a list of bigrams from each word and freqs of the word
        bigrams_in_word = [[char_in_word[j], char_in_word[j+1]] for j in range(len(char_in_word)-1)]
        freq = freqs[i]

        # store each bigram and its freq in the dics
        for idx in range(len(bigrams_in_word)):
            bigram = bigrams_in_word[idx]
            learn_key(bigram[0], bigram[1], freq) 


def get_probability(memory, freq_memory):
    # calculate the frequency of next chars after a certain char
    all_bigrams_dic = {}
    for key in memory.keys():
        # get all the second chars and their freqs following the key char
        values = memory[key]
        freqs = freq_memory[key]
        
        d = defaultdict(int)
        for i in range(len(values)):
            value = values[i]
            freq = freqs[i]
            d[value] += freq
        total_freqs = sum(freqs)

        # calculate the probability of each char following a certain char
        bi_prob = defaultdict(float)
        for value in values:
            bi_prob[value] = d[value]/total_freqs
            
        all_bigrams_dic[key] = bi_prob
    return all_bigrams_dic


# store all the first char from each word
def store_first_char(names):
    first_char_list = []
    for i in range(len(names)):
        word = names[i]
        char_in_word = word.split('_')
        first_char = char_in_word[0]
        first_char_list.append(first_char)
    return first_char_list  


# create a dis to store each first_char and its occurance
def first_char_dic(first_char_list):
    char_map = {}
    i = 1
    for j in range(len(first_char_list)):
        key = first_char_list[j]
        if key not in char_map:
            char_map[key] = i
        else:
            char_map[key] += 1  
    return char_map


def first_char_prob(first_char_list,char_map):
    total_sum = len(first_char_list)
    char_pro_map = dict(char_map)
    for key in char_pro_map.keys():
        value = char_pro_map[key]
        char_pro_map[key] = value/total_sum   
    return char_pro_map



def get_markov():
    # get a dictionary which has key characters and all the following characters with probability
    store_bigrams()
    all_prob = get_probability(memory, freq_memory) 

    # calculate the probabilities of all first characters
    first_chars = store_first_char(names)
    first_chars_dic = first_char_dic(first_chars)
    first_chars_prob = first_char_prob(first_chars, first_chars_dic)
    
    return all_prob, first_chars_prob
