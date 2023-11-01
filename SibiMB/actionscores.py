import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np

# Download the "text8" dataset
dataset = api.load("text8")

# Extract the data and create a word2vec model
# data = [dataset[0]]  # The data is stored in the first element of the dataset tuple
model = Word2Vec(dataset)


# List of physical actions, B words
B_word_cloud = ["running", "jumping", "swimming", "lifting", "climbing", "throwing", "crawling"]

# List of mental actions, C words
C_word_cloud = ["thinking", "imagining", "analyzing", "memorizing", "dreaming", "visualizing", "concentrating"]

# List of social actions, P words
P_word_cloud = ["talking", "listening", "helping", "sharing", "negotiating", "collaborating", "leading"]

# List of creative actions, S words
S_word_cloud = ["drawing", "painting", "writing", "singing", "dancing", "acting", "sculpting"]



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
# Find the Action scores

weights = {}
def findWeightedAnimalScore(actionevents, lattices): # currently the action events are inputted in entirely but for our purposes we need to do it action event by action event i am guessing somewhere in the pipeline
    for ae in actionevents:
        action = ae["action"]
        subject = ae["subject"]
        objects = ae["objects"]
        actionscores = [] # should end up with 4 action types

        # should be 4 lattices one for each action type
        for lattice_index in range(len(lattices)):
            lattice = lattices[lattice_index]
            action_score = []
            # lattice = nlp(lattice)
            # action = nlp(action)
            # number of words in each word cloud

            # how should we add the weights and initialize them randomly probably
            for word_index in range(len(lattice)):
                word = lattice[word_index]
                if word.has_vector() and action.has_vector():
                    action_score.append(weights[lattice_index][word_index] * action.similarity(word))
            avg_action_score = np.mean(action_score)
            actionscores.append(avg_action_score)
        ae["animals"] = actionscores
