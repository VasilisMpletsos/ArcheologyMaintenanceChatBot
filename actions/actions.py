# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# Imports for similarity search
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline, DPRQuestionEncoder, DPRContextEncoder
from sklearn.metrics.pairwise import cosine_similarity

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
    trust_remote_code=True,
    device_map="auto"
)
print("Loaded LLM model!")

# Load similarity model
similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
similarity_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded similarity model!")

# # Load question answer model
qa_model_name = "deepset/roberta-base-squad2"
qa_nlp = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)
print("Loaded QA model!")

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Read the extracted questions csv
df = pd.read_csv('./actions/cleared_questions.csv')
questions = df.question.to_list()
encoded_input = similarity_tokenizer(questions, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = similarity_model(**encoded_input)
# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = sentence_embeddings.detach().numpy()
print("Loaded extracted questions!")


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
        message = 'The restoration is recommended to be done the most until ' + recommended_date.strftime("%d/%m/%Y") + ' (d/m/y)'
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
        last_message = tracker.latest_message['text']
        # with open('unkown_intents.csv', 'a+', newline='') as f:
        #     f.write(last_message)
        #     f.write("\n")
        #     f.close()
        tokenized_query = similarity_tokenizer(last_message, padding=True, truncation=True, return_tensors='pt')
        embedded_query = similarity_model(**tokenized_query)
        question_embeddings = mean_pooling(embedded_query, tokenized_query['attention_mask'])
        question_embeddings = question_embeddings.detach().numpy()
        scores = cosine_similarity([question_embeddings[0]], sentence_embeddings)[0]
        max_pos = np.argmax(scores[1:])
        max_score = scores[max_pos+1]
        similar_question = questions[max_pos+1]
        context = df[df.question == similar_question].context.values[0]
        if max_score < 0.7:
            # dispatcher.utter_message('No answer found in the knowledge base')
            res = LLM(last_message)
            dispatcher.utter_message(res[0]['generated_text'])
        elif  max_score > 0.8 and max_score < 0.9:
            dispatcher.utter_message(f"Similar question found: {similar_question}")
            dispatcher.utter_message(f"Score: {max_score*100:.2f}%")
            dispatcher.utter_message('Answer found but low confidence')
            QA_input = {
                'question': last_message,
                'context': context
            }
            qa_result = qa_nlp(QA_input)
            dispatcher.utter_message(f"Answer: {qa_result['answer']}")
        else:
            dispatcher.utter_message(f"Similar question found: {similar_question}")
            dispatcher.utter_message(f"Score: {max_score*100:.2f}%")
            dispatcher.utter_message('Answer found from knowledge base with high confidence')
            dispatcher.utter_message(f"Answer: {context}")
        return