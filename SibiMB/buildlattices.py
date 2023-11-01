
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np

# Download the "text8" dataset
dataset = api.load("text8")

# Extract the data and create a word2vec model
# data = [dataset[0]]  # The data is stored in the first element of the dataset tuple
model = Word2Vec(dataset)


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