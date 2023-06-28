# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# Imports for similarity search and LLM
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
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

# ------------------------ INITIALLIZATION SECTION ------------------------

LLM = pipeline(
    model="databricks/dolly-v2-3b", 
    torch_dtype=torch.bfloat16,
    temperature=0.9, 
    trust_remote_code=True,
    device_map="auto",
    return_full_text=True
)

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=LLM)
llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
print("Loaded LLM model!")

# Load similarity model
similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
similarity_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded similarity model!")

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
print("Loaded knowledge base!")


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

            print(context)
            query = 'Answer the following question only with the provided input. ' + query;
            if max_score <= 0.50:
                dispatcher.utter_message("Sorry i don't know the answer to that question.")
            else:
                # dispatcher.utter_message("Similar context was found in the knowledgebase with high confidence, generating answer...")
                response = llm_context_chain.predict(instruction=query, context=context).lstrip()
                if response[-1] != '.':
                    response += '.'
                dispatcher.utter_message(response)
        else:
            dispatcher.utter_message("Please write complete questions to get an answer.")
            return