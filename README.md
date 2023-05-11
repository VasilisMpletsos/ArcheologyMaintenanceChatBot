# Archeology Maintenance Chat Bot

## Introduction
This is the github repository for the implementation of archeological preservation bot that is part of **Palimpsisto** project.
The bot is done on **Rasa** open source program. 

## Preparation
In order to be able to run the app you need to follow the rasa installation process to create an enviroment [rasa_env](https://rasa.com/docs/rasa/installation/environment-set-up) and to install dependencies [rasa_installation](https://rasa.com/docs/rasa/installation/installing-rasa-open-source). Next for the augmentations for paraphrasing you need to create a conda enviroment with all the necessary libraries.

## Augemntations
Because we didn't have data i proceed in the creation of a code that paraphrases the intents and the responses. Initially just place 3-4 examples under each intent and response.
After that place the required file that you want to get paraphrased under the need paraphrasing folder. Then run the paraphrasing script under the augmentations folder. Files for each intent and utterences will be generated in seperate txt files. Replace the old ones with the generated, make sure to check them and make any necessary changes.

## Out of scope
I will have to do somehting for out of scope, maybe save the phrases that were not understood in a txt file and maybe use a T5 model to answer them.