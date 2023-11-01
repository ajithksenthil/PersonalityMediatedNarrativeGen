import numpy as np
import openai
# import openai_secret_manager
import tkinter as tk
from tkinter import *
from tkinter import ttk
from io import BytesIO
from PIL import Image, ImageTk
import requests
import spacy
import re
# import torch
# import en_core_web_lg

import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

# Download the "text8" dataset
dataset = api.load("text8")

# Extract the data and create a word2vec model
# data = [dataset[0]]  # The data is stored in the first element of the dataset tuple
model = Word2Vec(dataset)

# model = api.load("book-corpus-large-cased")
# model = api.load("text8")

# model = gensim.models.Word2Vec.load("word2vec.model")
# nlp = en_core_web_lg.load()

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# global vars
first_time = True
current_state = {}
story = ""
behavior_states = {}

# Define the word clouds for each binary pair
B_word_cloud = ["Communicate", "Organize", "Teach", "Control", "Direct", "Start", "Brag", "Motivate", "Inspire",
                "Empower", "Lead", "Facilitate", "Guide", "Educate", "Mentor", "Encourage", "Advise", "Counsel",
                "Advocate", "Promote", "Endorse", "Support"]
C_word_cloud = ["Absorb", "Acquire", "Assemble", "Digest", "Enroll", "Gather", "Ingest", "Investigate", "Learn",
                "Peruse", "Receive", "Review", "Seek", "Study", "Take in", "Explore", "Respect", "Understand",
                "Analyze", "Comprehend", "Examine", "Scrutinize"]
P_word_cloud = ["Act", "Animate", "Cavort", "Compete", "Engage", "Entertain", "Frolic", "Gamble", "Game", "Jest",
                "Joke", "Leap", "Perform", "Prance", "Recreate", "Sport", "Toy", "Work", "Do", "Explore", "Show",
                "Teach", "Amuse", "Divert", "Enjoy", "Entertain"]
S_word_cloud = ["Catnap", "Doze", "Dream", "Hibernate", "Nap", "Nod", "Ponder", "Repose", "Rest", "Slumber", "Snooze",
                "Sustain", "Think", "Unwind", "Conserve", "Organize", "Introspect", "Process", "Preserve", "Meditate",
                "Reflect", "Relax", "Rejuvenate"]

b_word_cloud_lower = [word.lower() for word in B_word_cloud]
c_word_cloud_lower = [word.lower() for word in C_word_cloud]
p_word_cloud_lower = [word.lower() for word in P_word_cloud]
s_word_cloud_lower = [word.lower() for word in S_word_cloud]
# Convert the word clouds into word vectors using a semantic network
# B_vectors = [model.wv[word] for word in B_word_cloud if model.wv.get_vector(word) is not None]
# C_vectors = [model.wv[word] for word in C_word_cloud if model.wv.get_vector(word) is not None]
# P_vectors = [model.wv[word] for word in P_word_cloud if model.wv.get_vector(word) is not None]
# S_vectors = [model.wv[word] for word in S_word_cloud if model.wv.get_vector(word) is not None]

# B_vectors = [model.wv[word] for word in b_word_cloud_lower if model.wv.get_vector(word) is not None]
# C_vectors = [model.wv[word] for word in c_word_cloud_lower if model.wv.get_vector(word) is not None]
# P_vectors = [model.wv[word] for word in p_word_cloud_lower if model.wv.get_vector(word) is not None]
# S_vectors = [model.wv[word] for word in s_word_cloud_lower if model.wv.get_vector(word) is not None]

B_vectors = []
for word in b_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    B_vectors.append(vector)

C_vectors = []
for word in c_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    C_vectors.append(vector)

P_vectors = []
for word in p_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    P_vectors.append(vector)

S_vectors = []
for word in s_word_cloud_lower:
    try:
        vector = model.wv[word]
    except KeyError:
        # Word is not in the vocabulary, skip it
        continue
    S_vectors.append(vector)

# need to fill with optimized weights
weights = {}  # contains the weights for each word in the lattice provided by the initial word clouds

# get the action events

# list of dictionaries, each dictionary is an labeled action event with the action, subject, objects, and a key for the animals
# actionevents = []

characters = []


# Find the Animal scores after training with self supervision
# this is not used for initial prototype, it is for the optimized word clouds once trained with labeled data
def findWeightedAnimalScore(actionevents, lattices):
    for ae in actionevents:
        action = ae["action"]
        subject = ae["subject"]
        objects = ae["objects"]
        animals = []

        # should be 4 lattices one for each animal
        for lattice_index in range(len(lattices)):
            lattice = lattices[lattice_index]
            animal = []
            # lattice = nlp(lattice)
            # action = nlp(action)
            # number of words in each word cloud

            # how should we add the weights and initialize them randomly probably
            for word_index in range(len(lattice)):
                word = lattice[word_index]
                if word.has_vector() and action.has_vector():
                    animal.append(weights[lattice_index][word_index] * action.similarity(word))
            animal_score = np.mean(animal)
            animals.append(animal_score)
        ae["animals"] = animals


def gpt3(stext):
    openai.api_key = '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'
    response = openai.Completion.create(
        #        engine="davinci-instruct-beta",
        engine="text-davinci-003",
        prompt=stext,
        temperature=0.1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    content = response.choices[0].text.split('.')
    # print(content)
    return response.choices[0].text


def parse_output(output):
    print("parse output", output)
    tuples_list = []
    lines = output.split('\n')
    print("lines", lines)
    for line in lines:
        match = re.search(r"Action: (.+); Subject: (.+); Object: (.+)", line)
        print("match", match)
        if match:
            action = match.group(1)
            subject = match.group(2)
            object = match.group(3)
            tuples_list.append((action, subject, object))
    return tuples_list
    # for item in output:
    #     item = re.sub(r'\d+', '', item).strip()
    #     item = item.strip().replace('\n','')
    #     # Skip empty lines
    #     if not item:
    #         continue
    #     parts = item.split(', ')
    #     if len(parts) < 2:
    #         continue
    #     action = parts[0].split(': ')[1]
    #     subject = parts[1].split(': ')[1]
    #     if len(parts) > 2:
    #         object = parts[2].split(': ')[1]
    #     else:
    #         object = None
    #     tuples_list.append((action, subject, object))
    # print(tuples_list)
    # return tuples_list


def determine_next_action(story, characters, current_state, behavior_states):
    characters = characters
    # Use GPT to generate a list of likely next events based on the context of the story
    likely_events, actions, eventlistresp, subjectList = generate_likely_events(story, characters)
    print("likely_events", likely_events)
    print("for real actions", actions)
    # event_tuples_list = parse_output(likely_events)
    # event_tuples_list = likely_events
    eventlist = eventlistresp.split(",")
    # print("event_tuples_list", event_tuples_list)
    # actions = [x[0] for x in event_tuples_list]

    print("eventlist", eventlist)

    # Use the hidden Markov model to determine the likely next behavior state
    next_behavior_state = determine_next_behavior_state(current_state, likely_events, characters, behavior_states)

    # Use the semantic network to find the specific action that is most similar to the next behavior state
    next_action, actionindex = find_most_similar_action(next_behavior_state, characters, actions, subjectList)

    return next_action, actionindex, eventlist


# assert "openai" in openai_secret_manager.get_services()
# secrets = openai_secret_manager.get_secrets("openai")

# print(secrets)

# Use GPT to generate a list of likely next events based on the context of the story
def generate_likely_events(story, characters):
    characters = characters
    # Set up the GPT model
    storyInput = story
    query = f"Write a set of possible next events for a story, in which each event is a potential branch that could happen independently of the others. The events should not be sequential, but rather parallel paths that the following story could potentially take: {storyInput} with these characters: {characters}"
    response = gpt3(query)
    # print(response)
    query2 = f"For each of the following events, break it down into its component parts into a list and format them into a list in python: the action, the subject performing the action, and the object(s) affected by the action: {response}"
    response2 = gpt3(query2)
    print("response2", response2)

    queryEventList = f"Take this list of events {response} and format it to be one python string where each event is separated by commas like so 'event,event,event'"
    eventListResponse = gpt3(queryEventList)

    query3 = f"Using the provided list of events, please extract the single verb that represents the actions in each event and format them in the same order into a list in python like so 'verb, verb, verb, ...': {response2}"
    response3 = gpt3(query3)
    print("response3", type(response3))
    actions = response3.split(", ")
    actions = [s.strip() for s in actions]
    # actions = [i.splitlines()[0] for i in actions]
    print("actions 226", type(actions), actions)
    event_list = response2

    querySubj = f"Using the provided list of events, please extract the subject performing the action in each event and format them in the same order into a list in python like so 'subject, subject, subject, ...' here is the list of events: {response2}"
    subjectlist = gpt3(querySubj)
    subjectlist = subjectlist.strip().split(", ")
    print("subjectlist", subjectlist)

    return event_list, actions, eventListResponse, subjectlist


# Use the hidden Markov model for each character to determine the likely next behavior state
def determine_next_behavior_state(current_state, events, characters, behavior_states):
    next_behavior_states = {}
    print("characters", characters)

    for character in characters:
        # Get the current behavior state for the character
        current_behavior_state = current_state[character]

        # Use the hidden Markov model for the character to determine the likely next behavior state
        print("transition_probabilities", transition_probabilities)
        next_behavior_state = np.random.choice(behavior_states[character], p=transition_probabilities[character][
            getBehaviorIndex(current_behavior_state)])

        # Store the likely next behavior state for the character
        next_behavior_states[character] = next_behavior_state
        current_state[character] = next_behavior_state  # updating current states for each character
    print("next_behavior_states", next_behavior_states)

    return next_behavior_states


# Use the semantic network to find the specific action that is most similar to the next behavior state
# actions = [set of likely next events as action words]
def find_most_similar_action(next_behavior_state, characters, actions, subjectlist):
    # Implement the logic to use the semantic network to find the specific action that is most similar to the next behavior state
    # remove later
    next_behavior_state_one_character = next_behavior_state[characters[0]]
    # Initialize a dictionary to store the similarity scores for each action
    similarity_scores = {}
    print("actions", actions)
    for actionindex in range(len(actions)):
        action = actions[actionindex]
        try:
            next_behavior_state_one_character = next_behavior_state[subjectlist[actionindex]]
        except:
            print("subject did not match")
            next_behavior_state_one_character = next_behavior_state[characters[0]]

        print("action", action)
        infosimilarity = 0
        energysimilarity = 0
        try:
            action_vector = model.wv[action]
            action_vector = action_vector.reshape(1, -1)
            print("found action vector")
        except KeyError:
            # handle the exception (e.g. skip this iteration of the loop)
            print("did not find action vector")
            continue

        if next_behavior_state_one_character == "BS":
            infosimilarity = cosine_similarity(action_vector, B_vectors)
            energysimilarity = cosine_similarity(action_vector, S_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

        elif next_behavior_state_one_character == "BP":
            infosimilarity = cosine_similarity(action_vector, B_vectors)
            energysimilarity = cosine_similarity(action_vector, P_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

        elif next_behavior_state_one_character == "CS":
            infosimilarity = cosine_similarity(action_vector, C_vectors)
            energysimilarity = cosine_similarity(action_vector, S_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

        elif next_behavior_state_one_character == "CP":
            infosimilarity = cosine_similarity(action_vector, C_vectors)
            energysimilarity = cosine_similarity(action_vector, P_vectors)
            similarity_scores[action] = np.mean(infosimilarity) + np.mean(energysimilarity)

    # Find the action with the highest similarity score
    print("similarity scores", similarity_scores)

    # TODO use try and except statement and assign a random similarity score matrix the correct size.
    next_action = max(similarity_scores, key=similarity_scores.get)
    action_index = actions.index(next_action)
    print(next_action)
    return next_action, action_index


from hmmlearn import hmm


# train HMM
# modify this to use the predefined markov models
# Collect data on the behavior states of each character over time
# character_data = {
#   'Alice': ['BP', 'BS', 'CP', 'BP', 'BS', 'CP', 'BP'],
#   'Bob': ['BS', 'BP', 'CP', 'BS', 'BP', 'CP', 'BS']
# }

# # Preprocess the data to encode the behavior states as integers
# encoder = {}
# character_data_encoded = {}
# for character, data in character_data.items():
#   # Create an encoder to map the behavior states to integers
#   encoder[character] = {state: i for i, state in enumerate(set(data))}

#   # Encode the behavior states as integers
#   character_data_encoded[character] = [encoder[character][state] for state in data]

# # Define the set of possible behavior states and transition probabilities for each character
# # behavior_states = {
# #   'Alice': ['BP', 'BS', 'CP', 'CS'],
# #   'Bob': ['BP', 'BS', 'CP', 'CS']
# # }
# transition_probabilities = {
#   'Alice': np.array([[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.1, 0.9]]),
#   'Bob': np.array([[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.1, 0.9]])
# }

def initTransitionProbs(characters):
    global transition_probabilities
    transition_probabilities = {character: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for character in
                                characters}


# Train the hidden Markov model for each character
# character_models = {}
# for character, data in character_data_encoded.items():
#   # Create the hidden Markov model
#   model = hmm.MultinomialHMM(n_components=len(behavior_states[character]))

#   # Fit the model to the data using an iterative optimization algorithm
#   model.fit(np.array([data]).T, transition_probabilities=transition_probabilities[character])

#   # Store the trained model
#   character_models[character] = model

# hmm script portion end

# assign personalities to characters


personality_list = ["bpsc", "bpcs", "cbsp", "cpsb", "sbcp", "sbpc", "pcbs", "pcsb", "scbp", "scpb", "cbps", "csbp",
                    "bpsc", "bpsc", "spbc", "spcb"]
transition_matrices = {personality: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] for personality in
                       personality_list}

behavior_states_list = ['BP', 'BS', 'CP', 'CS']

# for personality in personality_list:
#     # Create a matrix of zeros
#     matrix = np.zeros((4,4))

#     # Fill in the matrix based on the personality
#     for i in range(4):
#         for j in range(2):
#             if personality[i] == str.lower(behavior_states[j]):
#                 matrix[i][j] = 0.8
#             else:
#                 matrix[i][j] = 0.2/(3)
#     transition_matrices[personality] = matrix

for personality in personality_list:
    # Create a matrix of zeros
    matrix = np.zeros((4, 4))

    # Fill in the matrix based on the personality
    if personality == "bpsc":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.1
        matrix[1][3] = 0.2
        matrix[2][0] = 0.2
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.1
        matrix[3][0] = 0.2
        matrix[3][1] = 0.1
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "bpcs":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "cbsp":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "sbcp":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "sbpc":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "pcbs":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "pcsb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "scbp":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "scpb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "pcbs":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "pcsb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "pcbs":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "scbp":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "scpb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "cbps":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "bpsc":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "spbc":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "spcb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "cspb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "cpsb":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    if personality == "csbp":
        matrix[0][0] = 0.6
        matrix[0][1] = 0.2
        matrix[0][2] = 0.1
        matrix[0][3] = 0.1
        matrix[1][0] = 0.1
        matrix[1][1] = 0.6
        matrix[1][2] = 0.2
        matrix[1][3] = 0.1
        matrix[2][0] = 0.1
        matrix[2][1] = 0.1
        matrix[2][2] = 0.6
        matrix[2][3] = 0.2
        matrix[3][0] = 0.1
        matrix[3][1] = 0.2
        matrix[3][2] = 0.1
        matrix[3][3] = 0.6
    transition_matrices[personality] = matrix


def assignPersonalitytoCharacter(character, personality):
    transition_probabilities[character] = transition_matrices[personality]


def update_story_data(action, actionindex, eventlist, story, characters):
    # eventrepresentation = eventlist[actionindex]
    event = eventlist[actionindex]
    # queryAction = f"Take this event representation: {eventrepresentation} and write a detailed narrative description that can be appended to continue this story: {story} with these characters: {characters}"
    queryAction = f"Take this event: {event} and concisely write the next event that can be appended to continue this story: {story} with these characters: {characters}"
    output = gpt3(queryAction)
    return output


def getBehaviorIndex(behaviorstate):
    if str.upper(behaviorstate) == 'BP':
        return 0
    elif str.upper(behaviorstate) == 'BS':
        return 1
    elif str.upper(behaviorstate) == 'CP':
        return 2
    elif str.upper(behaviorstate) == 'CS':
        return 3


def genSceneImage(actionevent, characters):
    queryScene = f"Show the scene of {actionevent} with characters {characters} in an creative and realistic way using DALL-E"
    imageprompt = gpt3(queryScene)
    return imageprompt


# import openai_secret_manager
# import openai
# import requests
# from PIL import Image

def genImagefromScene(imageprompt):

    # Get API key
    api_key = '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'

    # # Define the story
    # story = "Once upon a time, there was a young girl named Alice who went on a journey through a magical land. She met many interesting characters, such as a rabbit in a waistcoat and a caterpillar smoking a hookah. Along the way, she faced many challenges, but she always found a way to overcome them with her courage and determination."

    # Use the OpenAI API to generate the image
    openai.api_key = api_key
    prompt = (f"generate a image with Dall-E based on the following story: {imageprompt}")
    # prompt = imageprompt
    response = openai.Image.create(
        prompt=prompt,
        model="image-alpha-001"
    )

    # Get the image data from the URL
    response = requests.get(response["data"][0]["url"])
    img_data = Image.open(BytesIO(response.content))

    return img_data

    # Create a label to display the image
    # img = ImageTk.PhotoImage(img_data)
    # img_label = tk.Label(root, image=img)

    # # Authenticate to the OpenAI API
    # secrets = openai_secret_manager.get_secrets("openai")
    # openai.api_key = secrets["api_key"]

    # # Define the DALL-E prompt
    # prompt = imageprompt
    # # Use the prompt to generate an image with DALL-E
    # response = openai.Image.create(
    #     prompt=prompt,
    #     model="image-alpha-001"
    # )

    # # Get the URL of the generated image
    # image_url = response["data"]["url"]

    # # Download the image
    # response = requests.get(image_url)
    # img = Image.open(BytesIO(response.content))

    # # Save the image as a variable
    # image_variable = img

    # # You can also save the image to a file
    # img.save("scene.png")


# import openai

# make a userinterface for this application TODO

# import tkinter as tk
# from tkinter import *
# from tkinter import ttk
from tkinter import messagebox

# from PIL import ImageTk, Image

personality_list = ["bpsc", "bpcs", "cbsp", "cpsb", "sbcp", "sbpc", "pcbs", "pcsb", "scbp", "scpb", "cbps", "csbp",
                    "bpsc", "bpsc", "spbc", "spcb"]
story_image = True  # Just a placeholder, we need to build the image using PIL


class App:
    def __init__(self, story, characters, story_image):
        # Create the main window
        self.window = tk.Tk()
        self.window.title("SIBILANCE AI - STORY GENERATOR")
        # Changing the background color of the window
        self.window.configure(background='ghost white')

        self.lb1 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text=" CHARACTER ", bg="CadetBlue1", font=("Georgia, 14"))
        self.lb1.place(x=530, y=20)

        self.lb2 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text="   PERSONALITY     ", bg="CadetBlue1", font=("Georgia, 14"))
        self.lb2.place(x=670, y=20)

        self.tb1 = tk.Text(self.window, bd=5)
        self.tb1.place(x=530, y=60, height=30, width=110)

        self.cb1 = ttk.Combobox(self.window, width=5, font=("Georgia, 12"))
        self.cb1['values'] = (personality_list)
        self.cb1.place(x=670, y=60, width=150)

        self.tb2 = tk.Text(self.window, bd=5)
        self.tb2.place(x=530, y=100, height=30, width=110)

        self.cb2 = ttk.Combobox(self.window, width=5, font=("Georgia, 12"))
        self.cb2['values'] = (personality_list)
        self.cb2.place(x=670, y=100, width=150)

        self.tb3 = tk.Text(self.window, bd=5)
        self.tb3.place(x=530, y=140, height=30, width=110)

        self.cb3 = ttk.Combobox(self.window, width=5, font=("Georgia, 12"))
        self.cb3['values'] = (personality_list)
        self.cb3.place(x=670, y=140, width=150)

        self.lb6 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text="   STORY   ", bg="CadetBlue1", font=("Georgia, 14"))
        self.lb6.place(x=20, y=240)

        self.tb4 = tk.Text(self.window, bd=5)
        self.tb4.place(x=20, y=280, height=150, width=500)

        self.lb9 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, text="   ACTUAL IMAGE   ",
                            font=("Helvetica", 12), bg='#F0F0F0')
        self.lb9.place(x=20, y=20, height=200, width=500)

        self.lb10 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE,
                             text="   OPTIONALLY SUGGEST NEXT ACTION EVENT \n   WRITE A FULL MEANINGFUL SENTENCE   ",
                             bg="CadetBlue1", font=("Georgia, 13"))
        self.lb10.place(x=20, y=450)

        self.tb5 = tk.Text(self.window, bd=5)
        self.tb5.place(x=20, y=500, height=30, width=500)

        # self.btn1 = tk.Button(self.window, text="   DETERMINE NEXT ACTION EVENT   ", fg="black", font=("Georgia, 13"),
        #                       bg="gold", relief=tk.RAISED, command=self.getUIDataAndChooseAction)
        self.btn1 = tk.Button(self.window, text="   DETERMINE NEXT ACTION EVENT   ", fg="black", font=("Georgia, 13"),
                              bg="gold", command=self.getUIDataAndChooseAction)
        self.btn1.place(x=50, y=540)

        self.window.geometry("1000x600+10+10")
        self.window.mainloop()

    ##############################

    def getUIDataAndChooseAction(self):
        global first_time

        st_character1 = self.tb1.get("1.0", "end-1c")
        self.lb3 = tk.Label(self.window, bd=5, fg="#000", font=("Georgia, 12"), relief=tk.GROOVE, text=st_character1)
        self.lb3.place(x=530, y=60, height=30, width=110)

        st_character2 = self.tb2.get("1.0", "end-1c")
        self.lb4 = tk.Label(self.window, bd=5, fg="#000", font=("Georgia, 12"), relief=tk.GROOVE, text=st_character2)
        self.lb4.place(x=530, y=100, height=30, width=110)

        st_character3 = self.tb3.get("1.0", "end-1c")
        self.lb5 = tk.Label(self.window, bd=5, fg="#000", font=("Georgia, 12"), relief=tk.GROOVE, text=st_character3)
        self.lb5.place(x=530, y=140, height=30, width=110)

        characters = [st_character1, st_character2, st_character3]
        print(characters)

        st_personality_character1 = self.cb1.get()
        print(st_personality_character1)

        st_personality_character2 = self.cb2.get()
        print(st_personality_character2)

        st_personality_character3 = self.cb3.get()
        print(st_personality_character3)

        personality_characters = [st_personality_character1, st_personality_character2, st_personality_character3]
        print(personality_characters)

        # for character in characters:

        character_personality_tuple_list = self.createTuple(characters, personality_characters)
        print(character_personality_tuple_list)

        global story
        if first_time:
            story = self.tb4.get("1.0", "end-1c")
            print(story)
            # story = story = story.replace("\n", " ")
            # self.lb7 = tk.Label(self.window, bd=5, fg="#f0f", relief=tk.GROOVE, text=story)
            # self.lb7.place(x=20, y=280, height=150, width=500)

        user_suggested_next_action_event = self.tb5.get("1.0", "end-1c")
        user_suggestion = False
        if user_suggested_next_action_event != "":
            user_suggestion = True
        print(user_suggestion)
        print(user_suggested_next_action_event)

        # Use the following information coming from the UI and call appropriate functions
        # characters - this is a list
        # personality_characters - this is a tuple (char:personality) - The user could have changed the personality different from previous screen
        # story - free text with /n chars removed. Suggest whether we need to remove tab chars /t
        # user_suggestion - this is a flag - True or False
        # user_suggested_next_action_event - free text - this has actual suggested action event by the user_suggestion

        # Use the values gotten from your function modules to set the following values to display on the UI
        # characters = characters
        # story = story

        # story_image - this is a png image file for the story - very first screen, this will be blank - uncomment following two lines after setting up story_image value
        # self.lb9 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, image=story_image, font=("Helvetica", 12), bg='#F0F0F0')
        # self.lb9.place(x=320, y=20, height=200, width=450)
        # Add a Scrollbar(vertical)

        # Trying scrollable story - IT-3
        # v = Scrollbar(self.window, orient='vertical')
        # v.pack(side=RIGHT, fill='y')
        # v.destroy()

        if first_time:
            initTransitionProbs(characters=characters)
            global current_state
            # initialize current state randomly for now
            for character in characters:
                current_state[character] = random.choice(['BP', 'BS', 'CP', 'CS'])

            # initalize behavior state for all characters
            global behavior_states
            for character in characters:
                behavior_states[character] = ['BP', 'BS', 'CP', 'CS']

            # Trying scrollable story - IT-3
            # v = Scrollbar(self.window, orient='vertical')
            # v.pack(side=RIGHT, fill='y')
        first_time = False

        # assign the characters to the personalities picked
        for character, personality in character_personality_tuple_list:
            assignPersonalitytoCharacter(character, personality)

        next_action, actionindex, eventlist = determine_next_action(story=story, characters=characters,
                                                                    current_state=current_state,
                                                                    behavior_states=behavior_states)
        nextevent = update_story_data(next_action, actionindex=actionindex, eventlist=eventlist, story=story,
                                      characters=characters)

        # update the story with next event
        story = story + nextevent
        print("story ", story)
        print("nextevent:  ", nextevent)

        # self.lb7 = tk.Label(self.window, bd=5, fg="#f0f", padx=5, pady=5, relief=tk.GROOVE, text=story)
        # self.lb7.place(x=20, y=280, height=150, width=500)
        # win1=App(story, characters, story_image)

        # Trying scrollable story - IT-3
        # v = Scrollbar(self.window, orient='vertical')
        # v.pack(side=RIGHT, fill='y')

        # Add a text widget
        # text = Text(self.window, font=("Georgia, 12"), yscrollcommand=v.set)
        text = Text(self.window, font=("Georgia, 12"), padx=5, pady=5, selectborderwidth=50, wrap=tk.WORD)
        text.insert(END, story)

        # Attach the scrollbar with the text widget
        # v.config(command=text.yview)
        text.place(x=20, y=280, height=150, width=500)

        # Trying scrollable story - IT-3 - End

        # Get the image for nextevent and display in a label lb9
        # Start
        imageprompt = genSceneImage(nextevent, characters)
        # Split imageprompt into sentences
        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', imageprompt)
        print("imageprint: " + imageprompt)
        print("sentences[0]")
        print(sentences[0])
        #
        img_data = genImagefromScene(sentences[0])

        # Add the image to the label
        img = ImageTk.PhotoImage(img_data)

        # image = Image.open(img)
        image = img_data.resize((500, 200))

        photo = ImageTk.PhotoImage(image)

        # self.lb9 = tk.Label(self.window, bd=5, fg="#00f", relief=tk.GROOVE, image=img, font=("Helvetica", 12), bg='#F0F0F0')
        self.lb9 = tk.Label(self.window, image=photo)
        self.lb9.place(x=20, y=20, height=200, width=500)

        # End

        return

    # Utility Function - To create the tuples using two lists
    def createTuple(self, list1, list2):

        tuple_list = []
        for i in range(max((len(list1), len(list2)))):

            while True:
                try:
                    tup = (list1[i], list2[i])
                except IndexError:
                    if len(list1) > len(list2):
                        list2.append('')
                        tup = (list1[i], list2[i])
                    elif len(list1) < len(list2):
                        list1.append('')
                        tup = (list1[i], list2[i])
                    continue

                tuple_list.append(tup)
                break
        return tuple_list

    # def findNextActionEvent(self):
    #     st_character1 = self.tb1.get("1.0", "end-1c")
    #     self.lb3 = tk.Label(self.window, bd=5, fg="#000", relief=tk.GROOVE, text=st_character1)
    #     self.lb3.place(x=20, y=60, height=30, width=110)
    #
    #     st_character2 = self.tb2.get("1.0", "end-1c")
    #     self.lb4 = tk.Label(self.window, bd=5, fg="#000", relief=tk.GROOVE, text=st_character2)
    #     self.lb4.place(x=20, y=100, height=30, width=110)
    #
    #     st_character3 = self.tb3.get("1.0", "end-1c")
    #     self.lb5 = tk.Label(self.window, bd=5, fg="#000", relief=tk.GROOVE, text=st_character3)
    #     self.lb5.place(x=20, y=140, height=30, width=110)
    #
    #     characters = [st_character1, st_character2, st_character3]
    #     print(characters)
    #
    #     # BUILD AN ARRAY OF TUPLES OF CHARACTER AND PERSONALITY
    #
    #     # CAPTURE USER ENTERED STORY IN THE VERY FIRST TIME AND THEN JUST SEND THE STORY TO AJJU FN MODULE
    #     story = self.tb4.get("1.0", "end-1c")
    #     print(story)
    #
    #     #     sentence = self.txt1.get(1.0, "end-1c")
    #     #     word_list = sentence.split()
    #     #     story = self.txt21.get(1.0, "end")
    #     #     story1=story
    #     #     #story = story.replace("\n", " ")
    #     #     characters = word_list
    #     #
    #     #     self.lbl4=tk.Label(self.window, bd=5, relief=tk.RIDGE, justify=tk.LEFT, text=story)
    #     #     self.lbl4.place(x=140, y=60, height=150, width=500)
    #     #     self.lbl4["text"] = story1
    #     #     self.cb1["values"] = characters
    #     #
    #     #     print(sentence)
    #     #     print(word_list)
    #     #     print(word_list[1])
    #     #     print(story)
    #     return


import random


def main():
    # TODO

    current_state = {}
    # current_state = {'Alice': 'BP','Bob': 'CS'}
    behavior_states = {}
    # behavior_states = {'Alice': ['BP', 'BS', 'CP', 'CS'],'Bob': ['BP', 'BS', 'CP', 'CS']}
    # story = "Alice and Bob are going to a grocery store."
    characters = []
    story = ""
    story_image = ""  # supposed to be a png file from DallE
    setting = ""
    genre = ""
    worldbuilding = {"setting": setting, "genre": genre}

    win1 = App(story, characters, story_image)  # UI
    # if first_time:
    #   # initialize current state randomly for now
    #   for character in characters:
    #     current_state[character] = random.choice(['BP', 'BS', 'CP', 'CS'])

    #   # initalize behavior state for all characters
    #   for character in characters:
    #     behavior_states[character] = ['BP', 'BS', 'CP', 'CS']
    # first_time = False

    # next_action, actionindex, eventlist = determine_next_action(story="Alice and Bob are going to a grocery store.", characters=["Alice", "Bob"], current_state=current_state, behavior_states=behavior_states)
    # nextevent = update_story_data(next_action, actionindex=actionindex, eventlist=eventlist, story=story, characters=characters)

    # # update the story with next event
    # story = story + nextevent
    # win1=App(story, characters, story_image)

    # print("eventlist", eventlist)
    # print("next_action", next_action)
    # print("action_index", actionindex)
    # print("nextevent", nextevent)


if __name__ == "__main__":
    main()



