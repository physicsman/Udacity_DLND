import os
import pickle
import re
import pandas
import numpy as np


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

# punctuation = '([' + string.punctuation +'])' # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
punctuation = '([.,";!?()])' # Define our punctuation

def add_token_spaces(text):
    # Pad punctuation with whitespace
    tokens = ['--', '\n']
    tokens_sub = ['juxqz', 'yunfr']
    for n in range(len(tokens)): # Token subsitution
        text = text.replace(tokens[n], tokens_sub[n])
    text = re.sub(punctuation, r' \1 ', text)
    tokens = [' -- ', ' \\n ']
    for n in range(len(tokens)): # Token de-subsitution
        text = text.replace(tokens_sub[n], tokens[n]) 
    text = re.sub(punctuation, r' \1 ', text)
    
    return text

def remove_token_spaces(tv_script):
    tv_script = tv_script.replace('\\n', '\n')
    tv_script = tv_script.replace('( ', '(')
    for n in range(len(punctuation)):
        p = punctuation[n]
        tv_script = tv_script.replace(' ' + p, p)
    
    return tv_script

#def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
def preprocess_and_save_data(text, scene_marker, create_lookup_tables):
    """
    Preprocess Text Data
    """
    data = []
    words = []
    for n in range(len(text)):
        line = [ '\n' + text[n,0] + ':']
        txt = text[n,1]
        txt = add_token_spaces(txt)
        txt = txt.lower().split()
        line.extend(txt)
        data.append(line)
        words.extend(line)
    
    vocab_to_int, int_to_vocab = create_lookup_tables(words)
    for n in range(len(data)):
        data[n] = [vocab_to_int[word] for word in data[n]]
    # pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))
    pickle.dump((data, scene_marker, vocab_to_int, int_to_vocab), open('preprocess_new.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess_new.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))

def save_params_jj(params, file):
    """
    Save parameters to file
    """
    pickle.dump(params, open(file, 'wb'))
    
def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))

def load_params_jj(file):
    """
    Load parameters from file
    """
    return pickle.load(open(file, mode='rb'))

def create_simpsons(file):
    data_dir = './data/simpsons/simpsons_script_lines_sorted.csv'
    script_lines = pandas.read_csv(data_dir, warn_bad_lines=True, error_bad_lines=False)
    
    word_count = script_lines.loc[:,'word_count'].values
    ind = []
    for n in range(len(word_count)):
        val = word_count[n]
        try:
            int(val)
            ind.append(n)
        except:
            pass
    word_count = word_count[ind].astype(np.int32)
    episode_id = script_lines.loc[:,'episode_id'].fillna(0).values
    episode_id = episode_id[ind].astype(np.int32)
    character_id = script_lines.loc[:,'character_id'].fillna(0).values
    character_id = character_id[ind].astype(np.int32)
    raw_character_text = script_lines.loc[:,'raw_character_text'].values
    raw_character_text = raw_character_text[ind]
    location_id = script_lines.loc[:,'location_id'].fillna(0).values
    location_id = location_id[ind].astype(np.int32)
    raw_location_text = script_lines.loc[:,'raw_location_text'].values
    raw_location_text = raw_location_text[ind]
    spoken_words = script_lines.loc[:,'spoken_words'].values
    spoken_words = spoken_words[ind]
    int_to_character = {}
    character_to_int = {}
    for n in range(len(raw_character_text)):
        c = raw_character_text[n]
        if c not in int_to_character.values():
            int_to_character[character_id[n]] = c
            character_to_int[c] = character_id[n]
            
    save_params_jj((word_count, episode_id,
                    character_id, raw_character_text,
                    location_id, raw_location_text,
                    spoken_words,
                    int_to_character, character_to_int),file)
