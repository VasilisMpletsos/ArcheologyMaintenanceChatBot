# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# Imports for similarity search and LLM
import os
import re
import random
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline, PreTrainedTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# Imports for date recommendation
import datetime
from random import randint, choice
# Rasa imports
from typing import Any, Text, Dict, List
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import (
    SlotSet,
    EventType,
)



# ------------------------ LLM NECESSARY CODE ------------------------

# Prompt
INTRO_BLURB = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.")

# To be added as special tokens
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"


PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT_WITH_INPUT = """{intro}

{instruction_key}
{instruction}

{input_key}
{context}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    context="{context}",
    response_key=RESPONSE_KEY,
)

def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        print(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]

def preprocess(tokenizer, instruction_text, context_text=None):
    instruction = "Answer the following question only with the provided input. If no answer is found tell that you cannot answer based on this context. " + instruction_text
    if context_text:
        prompt_text = PROMPT_FOR_GENERATION_FORMAT_WITH_INPUT.format(instruction=instruction, context=context_text)
    else:
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)

    inputs = tokenizer(prompt_text, return_tensors="pt",)
    inputs["prompt_text"] = prompt_text
    inputs["instruction_text"] = instruction_text
    inputs["context_text"] = context_text
    return inputs

def forward(model, tokenizer, model_inputs, max_length=256):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]

    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length
    )

    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    instruction_text = model_inputs.pop("instruction_text", None)

    return {
        "generated_sequence": generated_sequence, 
        "input_ids": input_ids,
    }


def postprocess(tokenizer, model_outputs):
    response_key_token_id = get_special_token_id(tokenizer, RESPONSE_KEY_NL)
    end_key_token_id = get_special_token_id(tokenizer, END_KEY)
    generated_sequence = model_outputs["generated_sequence"][0]
    
    # send it to cpu
    generated_sequence = generated_sequence.cpu()
    generated_sequence = generated_sequence.numpy().tolist()
    records = []

    for sequence in generated_sequence:
        decoded = None

        try:
            response_pos = sequence.index(response_key_token_id)
        except ValueError:
            print(f"Could not find response key {response_key_token_id} in: {sequence}")
            response_pos = None

        if response_pos:
            try:
                end_pos = sequence.index(end_key_token_id)
            except ValueError:
                print("Could not find end key, the output is truncated!")
                end_pos = None
                
            if end_pos:
                decoded = tokenizer.decode(sequence[response_pos + 1 : end_pos], skip_special_tokens=True).strip()
            else:
                decoded = "Sorry i cannot answer this question";         
            
        rec = {"generated_text": decoded}
        records.append(rec)
    return records

def get_model_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path);
    model = model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype = torch.bfloat16,
    );
    model.resize_token_embeddings(len(tokenizer));
    return model, tokenizer

def clean_text(text):
    # strip sentenece
    text = text.strip()
    # remove tabs
    text = text.replace('\t', '')
    # remove new lines
    text = text.replace('\n', '')
    return text

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ------------------------ INITIALLIZATION SECTION ------------------------


# Load the LLM Dolly v2 3b model with its tokenizer
LLM_model, LLM_tokenizer = get_model_tokenizer(pretrained_model_name_or_path = "./scripts/FineTunedDollyV2");
LLM_model = LLM_model.to('cuda');
print('INFO:     Loaded LLM Model');

model_name = "deepset/roberta-base-squad2"
qa = pipeline('question-answering', model=model_name, tokenizer=model_name)
print('INFO:     Loaded QA Model')

# Load similarity model
similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
similarity_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("INFO:     Loaded similarity model!")

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Read the extracted questions csv
df = pd.read_csv('./actions/new_passages.csv')
passages = df.passages.to_list()
encoded_input = similarity_tokenizer(passages, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = similarity_model(**encoded_input)
# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = sentence_embeddings.detach().numpy()
print("INFO:     Loaded knowledge base!")

unanswerable_questions = [
    "I do not know the answer to this question.",
    "Sorry i cannot answer this question.",
    "Sadly i can't answer this question.",
    "I don't know the answer to this question.",
    "I don't know the answer to this question, sorry.",
    "I do not have the answer to this question.",
    "I don't possess the answer to this question.",
    "Sorry i am not capable of answering this question.",
    "Sorry i cannot answer",
    "It appears that i cannot answer this question.",
]
# ------------------------ CLASSES FOR RASA ------------------------
    
class ActionEvaluateSlowDegradationRate(Action):
    """Respond to slow degradation rate"""

    def name(self) -> Text:
        return "action_recommend_for_slow_degradation_rate"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        today = datetime.datetime.now()
        random_delta = randint(90, 180)
        recommended_date = today + datetime.timedelta(days=random_delta)
        message = 'The restoration is recommended to be done until somewhere ' + recommended_date.strftime("%d/%m/%Y") + ' (d/m/y)'

        dispatcher.utter_message(message)
        return
    
class ActionEvaluateMediumDegradationRate(Action):
    """Respond to medium degradation rate"""

    def name(self) -> Text:
        return "action_recommend_for_medium_degradation_rate"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        today = datetime.datetime.now()
        random_delta = randint(30, 90)
        recommended_date = today + datetime.timedelta(days=random_delta)
        message = 'The restoration is recommended to be done utmost until ' + recommended_date.strftime("%d/%m/%Y") + ' (d/m/y)'
        dispatcher.utter_message(message)
        return
    
class ActionEvaluateFastDegradationRate(Action):
    """Respond to fast degradation rate"""

    def name(self) -> Text:
        return "action_recommend_for_fast_degradation_rate"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        today = datetime.datetime.now()
        random_delta = randint(10, 30)
        recommended_date = today + datetime.timedelta(days=random_delta)
        message = 'The restoration is recommended to be done as soon as possible until ' + recommended_date.strftime("%d/%m/%Y") + ' (d/m/y)'
        dispatcher.utter_message(message)
        return
    
class ActionSaveUnkownIntent(Action):
    """Reply to the unknown intent"""

    def name(self) -> Text:
        return "action_reply_unknown_intent"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        
        query = tracker.latest_message['text']
        if len(query) > 4:
            # with open('unkown_intents.csv', 'a+', newline='') as f:
            #  The query here has some error and cant write all the charachters need to see why.
            #     f.write(query)
            #     f.write("\n")
            #     f.close()
            tokenized_query = similarity_tokenizer(query, padding=True, truncation=True, return_tensors='pt')
            embedded_query = similarity_model(**tokenized_query)
            question_embeddings = mean_pooling(embedded_query, tokenized_query['attention_mask'])
            question_embeddings = question_embeddings.detach().numpy()
            scores = cosine_similarity([question_embeddings[0]], sentence_embeddings)[0]
            max_pos = np.argmax(scores[1:])
            max_score = scores[max_pos+1]
            context = passages[max_pos+1]
            
            # Calculate the answer of the QA model
            QA_input = {
                'question': query,
                'context': context
            }
            res = qa(QA_input)
            answer_qa = res['answer']

            query = 'Answer the following question only with the provided input. ' + query;
            if max_score > 0.4:
                if res['score'] > 0.4:
                    # If QA model is confident then return its answer
                    dispatcher.utter_message(answer_qa);
                else:
                    # dispatcher.utter_message("Similar context was found in the knowledgebase with high confidence, generating answer...")
                    pre_process_result = preprocess(LLM_tokenizer, query, context);
                    model_result = forward(LLM_model, LLM_tokenizer, pre_process_result);
                    final_output = postprocess(LLM_tokenizer, model_result);
                    response = final_output[0]['generated_text'];
                    if response[-1] != '.':
                        response += '.'
                    dispatcher.utter_message(response);
            else:
                dispatcher.utter_message(random.choice(unanswerable_questions));
        else:
            dispatcher.utter_message("Please write complete questions to get an answer.");
            return