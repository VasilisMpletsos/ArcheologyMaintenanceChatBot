# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


import datetime
import os
from random import randint, choice
from typing import Any, Text, Dict, List
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import (
    SlotSet,
    EventType,
)

    
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
    """Save the unknown intent"""

    def name(self) -> Text:
        return "action_save_unknown_intent"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        last_message = tracker.latest_message['text']
        with open('unkown_intents.csv', 'a+', newline='') as f:
            f.write(last_message)
            f.write("\n")
            f.close()
        return