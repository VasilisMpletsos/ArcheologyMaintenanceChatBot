# Include the necessary libraries
import torch
import warnings
import yaml
import os

# This one is for paraphrasing
from parrot import Parrot

# suppress warnings
warnings.filterwarnings("ignore")

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True)

# read a yaml file
def read_yaml_file(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
# clean phrases
def clean_phrases(examples):
    phrases = examples.split('\n')
    # remove empty strings
    phrases = [phrase for phrase in phrases if phrase != '']
    # remove dash and space
    phrases = [phrase[2:] for phrase in phrases]
    return phrases

# create paraphrased folder
if not os.path.exists('../data/nlu/paraphrased'):
    os.makedirs('../data/nlu/paraphrased')

# iterate over the nlu files
for file in os.listdir('../data/nlu/needs_paraphrasing'):
    # if file is a yaml file
    if file.endswith('.yml'):
        # print the file name
        print(f'Looking at {file}')
        # read yaml file
        data = read_yaml_file('../data/nlu/needs_paraphrasing/' + file)
        # file name
        file_name = file.split('.')[0]
        if not os.path.exists(f'../data/nlu/paraphrased/{file_name}'):
            os.makedirs(f'../data/nlu/paraphrased/{file_name}')
        # for each intent in nlu
        for row in data['nlu']:
            intent = row['intent']
            examples = row['examples']
            # get phrases
            phrases = clean_phrases(examples)
            initial_plen = len(phrases)
            final_phrases = []
            for phrase in phrases:
                para_phrases = parrot.augment(input_phrase=phrase)
                final_phrases.append(phrase)
                for para_phrase in para_phrases:
                    # if score is bigger than 10 append the paraphrased phrase to the list
                    if para_phrase[1] > 10:
                        # append the paraphrased phrase to the list
                        final_phrases.append(para_phrase[0])
            final_len = len(final_phrases)
            # print the number of paraphrased phrases
            print(f"Number of paraphrased phrases for {intent}: {final_len - initial_plen}")
            final_phrases = ['- ' + phrase for phrase in final_phrases]
            updated_phrases = '\n'.join(final_phrases)
            # write to data.txt
            with open(f"../data/nlu/paraphrased/{file_name}/{intent}.txt", 'w') as f:
                f.write(updated_phrases)
        
# delete the needs_paraphrasing files
for file in os.listdir('../data/nlu/needs_paraphrasing'):
    os.remove(f'../data/nlu/needs_paraphrasing/{file}')