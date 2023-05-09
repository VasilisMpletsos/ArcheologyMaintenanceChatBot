# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import datetime
from random import randint, choice
from typing import Any, Text, Dict, List
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import (
    SlotSet,
    EventType,
)

class ActionSetStructureMarbleSlot(Action):
    """Set the marble slot structure"""

    def name(self) -> Text:
        return "action_set_structure_marble"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        dispatcher.utter_message(response="utter_confirm_marble_material")
        return [SlotSet("structure_marble", True), SlotSet("structure_stone", False), SlotSet("structure_mortar", False), SlotSet("structure_shale", False), SlotSet("structure_conch_shells", False)]
    
class ActionSetStructureStoneSlot(Action):
    """Set the stone slot structure"""

    def name(self) -> Text:
        return "action_set_structure_stone"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        dispatcher.utter_message(response="utter_confirm_stone_material")
        return [SlotSet("structure_marble", False), SlotSet("structure_stone", True), SlotSet("structure_mortar", False), SlotSet("structure_shale", False), SlotSet("structure_conch_shells", False)]
    
class ActionSetStructureMortarSlot(Action):
    """Set the mortar slot structure"""

    def name(self) -> Text:
        return "action_set_structure_mortar"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        dispatcher.utter_message(response="utter_confirm_mortar_material")
        return [SlotSet("structure_marble", False), SlotSet("structure_stone", False), SlotSet("structure_mortar", True), SlotSet("structure_shale", False), SlotSet("structure_conch_shells", False)]
    
class ActionSetStructureShaleSlot(Action):
    """Set the shale slot structure"""

    def name(self) -> Text:
        return "action_set_structure_shale"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        dispatcher.utter_message(response="utter_confirm_shale_material")
        return [SlotSet("structure_marble", False), SlotSet("structure_stone", False), SlotSet("structure_mortar", False), SlotSet("structure_shale", True), SlotSet("structure_conch_shells", False)]
    
class ActionSetStructureConchSlot(Action):
    """Set the shale conch shells structure"""

    def name(self) -> Text:
        return "action_set_structure_conch_shells"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        dispatcher.utter_message(response="utter_confirm_conch_shells_material")
        return [SlotSet("structure_marble", False), SlotSet("structure_stone", False), SlotSet("structure_mortar", False), SlotSet("structure_shale", False), SlotSet("structure_conch_shells", True)]
    
class ActionEvaluateSlowDegradationRate(Action):
    """Respond to slow degredation rate"""

    def name(self) -> Text:
        return "action_recommend_for_slow_degradation_rate"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        today = datetime.datetime.now()
        random_delta = randint(30, 180)
        recommended_date = today + datetime.timedelta(days=random_delta)
        message = 'Η συντήρηση να γίνει μέχρι τις ' + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_marble"):
        #     message = choice(marble_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_stone"):
        #     message = choice(stone_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_mortar"):
        #     message = choice(mortar_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_shale"):
        #     message = choice(shale_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_conch_shells"):
        #     message = choice(conch_shells_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        dispatcher.utter_message(message)
        return
    
class ActionEvaluateMediumDegradationRate(Action):
    """Respond to medium degredation rate"""

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
        message = 'Η συντήρηση να γίνει μέχρι τις ' + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_marble"):
        #     message = choice(marble_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_stone"):
        #     message = choice(stone_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_mortar"):
        #     message = choice(mortar_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_shale"):
        #     message = choice(shale_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_conch_shells"):
        #     message = choice(conch_shells_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        dispatcher.utter_message(message)
        return
    
class ActionEvaluateFastDegradationRate(Action):
    """Respond to fast degredation rate"""

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
        message = 'Η συντήρηση να γίνει μέχρι τις ' + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_marble"):
        #     message = choice(marble_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_stone"):
        #     message = choice(stone_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_mortar"):
        #     message = choice(mortar_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_shale"):
        #     message = choice(shale_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        # if tracker.get_slot("structure_conch_shells"):
        #     message = choice(conch_shells_slow_rate) + recommended_date.strftime("%d/%m/%Y")
        dispatcher.utter_message(message)
        return