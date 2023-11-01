# SemanticLattice created by Ajith Senthil May 19 2022

# word clouds create the lattices we need four lattices for each animal 
# and then we have to find the animal scores for each action event which we use as our personality metric

# import necessary modules below

import spacy
import numpy as np
import torch


# load spacy import semantic network
import en_core_web_lg
nlp = en_core_web_lg.load()
# nlp = spacy.load('en_core_web_md')

# download the word clouds 

wordclouds = []



# construct the lattice from the word clouds 

# blast/Directing = 0, sleep = 1, play = 2, consume = 3
lattices = []
# need to fill
weights = {} # contains the weights for each word in the lattice provided by the initial word clouds
for animal in wordclouds:
    lattice = []
    for word_index in range(len(animal)):
        # weight = weights[animal][word_index]
        word = animal[word_index]
        lattice.append(word)
    lattices.append(lattice)
    

# get the action events

# list of dictionaries, each dictionary is an labeled action event with the action, subject, objects, and a key for the animals
actionevents = []

# get the transformation vectors from labeled data set

step_size = 1 # masculine and feminine metric

for ae in actionevents:
    action = ae["action"]
    subject = ae["subject"]
    objects = ae["objects"]
    animals = []
    labeledAnimal = ae[labeledAnimal]

    # should be 4 lattices one for each animal
    for lattice_index in range(len(lattices)):
        lattice = lattices[lattice_index]
        animal = [] # action choice/personality/behavior metric
        lattice = nlp(lattice)
        action = nlp(action)
        # number of words in each word cloud
        # how should we add the weights and initialize them randomly probably
        for word_index in range(len(lattice)):
            word = lattice[word_index]

            if word.has_vector() and action.has_vector():
                if labeledAnimal == lattice_index:
                    weights[lattice_index][word_index] += (1/(1 + np.exp((word.vector - action.vector))))*step_size
                else:
                    weights[lattice_index][word_index] -= 1/(1/(1 + np.exp((word.vector - action.vector))))*step_size







# Find the Animal scores after training with self supervision

for ae in actionevents:
    action = ae["action"]
    subject = ae["subject"]
    objects = ae["objects"]
    animals = []

    # should be 4 lattices one for each animal
    for lattice_index in range(len(lattices)):
        lattice = lattices[lattice_index]
        animal = []
        lattice = nlp(lattice)
        action = nlp(action)
        # number of words in each word cloud

        # how should we add the weights and initialize them randomly probably
        for word_index in range(len(lattice)):
            word = lattice[word_index]
            if word.has_vector() and action.has_vector():
                animal.append(weights[lattice_index][word_index]*action.similarity(word))
        animal_score = np.mean(animal)
        animals.append(animal_score)
    ae["animals"] = animals

    
    







# for testing can remove later
print("Enter two space-separated words")
words = input()
  
tokens = nlp(words)
  
for token in tokens:
    # Printing the following attributes of each token.
    # text: the word string, has_vector: if it contains
    # a vector representation in the model, 
    # vector_norm: the algebraic norm of the vector,
    # is_oov: if the word is out of vocabulary.
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
  
token1, token2 = tokens[0], tokens[1]
  
print("Similarity:", token1.similarity(token2))
