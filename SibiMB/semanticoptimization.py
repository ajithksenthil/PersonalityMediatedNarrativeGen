import gensim
import numpy as np
import spacy 
from scipy.spatial.distance import cdist



# load spacy import semantic network
import en_core_web_lg
nlp = en_core_web_lg.load()


# Load the word2vec model
model = gensim.models.Word2Vec.load('word2vec_model.bin')

# The list of action words
actions = [('Talk', [1, 1]),
 ('Teach', [1, 1]),
 ('Present', [1, 1]),
 ('Lead', [1, 1]),
 ('Show', [1, 1]),
 ('Demonstrate', [1, 1]),
 ('Prepare', [1, 0]),
 ('Plan', [1, 0]),
 ('Brainstorm', [1, 0]),
 ('Review', [1, 0]),
 ('Learn', [0, 1]),
 ('Attend', [0, 1]),
 ('Participate', [0, 1]),
 ('Ask', [0, 1]),
 ('Experiment', [0, 1]),
 ('Review', [0, 0]),
 ('Write', [0, 0]),
 ('Analyze', [0, 0]),
 ('Reflect', [0, 0]),
 ('Study', [0, 0]),
 ('Read', [0, 0]),
 ('Write', [0, 0]),
 ('Research', [0, 0]),
 ('Observe', [0, 0]),
 ('Listen', [0, 0]),
 ('Watch', [0, 0]),
 ('Think', [0, 0]),
 ('Talk', [1, 1]),
 ('Communicate', [1, 1]),
 ('Share', [1, 1]),
 ('Explain', [1, 1]),
 ('Discuss', [1, 1]),
 ('Debate', [1, 1]),
 ('Argue', [1, 1]),
 ('Persuade', [1, 1]),
 ('Negotiate', [1, 1]),
 ('Sell', [1, 1]),
 ('Advertise', [1, 1]),
 ('Promote', [1, 1]),
 ('Brand', [1, 1]),
 ('Market', [1, 1]),
 ('Publicize', [1, 1]),
 ('Educate', [1, 1]),
 ('Instruct', [1,1]),
 ('Train', [1,1]),
 ('Coach', [1,1]),
 ('Mentor', [1,1]),
 ('Counsel', [1,1]),
 ('Advise', [1,1]),
 ('Guide', [1,1])]

# The four word clouds
b_word_cloud_1 = ["Communicate", "Organize", "Teach", "Control", "Direct", "Start", "Brag", "Motivate", "Inspire", "Empower", "Lead", "Facilitate", "Guide", "Educate", "Mentor", "Encourage", "Advise", "Counsel", "Advocate", "Promote", "Endorse", "Support"]
c_word_cloud_2 =  ["Absorb", "Acquire", "Assemble", "Digest", "Enroll", "Gather", "Ingest", "Investigate", "Learn", "Peruse", "Receive", "Review", "Seek", "Study", "Take in", "Explore", "Respect", "Understand", "Analyze", "Comprehend", "Examine", "Scrutinize"]
p_word_cloud_3 = ["Act", "Animate", "Cavort", "Compete", "Engage", "Entertain", "Frolic", "Gamble", "Game", "Jest", "Joke", "Leap", "Perform", "Prance", "Recreate", "Sport", "Toy", "Work", "Do", "Explore", "Show", "Teach", "Amuse", "Divert", "Enjoy", "Entertain"]
s_word_cloud_4 = ["Catnap", "Doze", "Dream", "Hibernate", "Nap", "Nod", "Ponder", "Repose", "Rest", "Slumber", "Snooze", "Sustain", "Think", "Unwind", "Conserve", "Organize", "Introspect", "Process", "Preserve", "Meditate", "Reflect", "Relax", "Rejuvenate"]

# Find the semantic word vectors for each word in each word cloud
b_vectors_1 = [model[word] for word in b_word_cloud_1]
c_vectors_2 = [model[word] for word in c_word_cloud_2]
p_vectors_3 = [model[word] for word in p_word_cloud_3]
s_vectors_4 = [model[word] for word in s_word_cloud_4]

info_vectorlist = [b_vectors_1, c_vectors_2]
energy_vectorlist = [p_vectors_3, s_vectors_4]



# find the weighted transformations
info_weights = [[]]
energy_weights = [[]]

step_size = 1
# actions = [(action, (info, energy)), ...]
for action in actions:
    
    for information_index in range(len(info_vectorlist)):
        lattice = info_vectorlist[information_index]
        animal = [] # action choice/personality/behavior metric
        lattice = nlp(lattice)
        # number of words in each word cloud
        # how should we add the weights and initialize them randomly probably
        for word_index in range(len(lattice)):
            word = lattice[word_index]
            labeledAnimalPair = action[1] # (info (either B(1) or C(0)), energy(either P(1) or S(0)))
            if action[0].has_vector():
                
                # information animal 
                if labeledAnimalPair[0] == information_index:
                    info_weights[information_index][word_index] += (1/(1 + np.exp((word.vector - action[0].vector))))*step_size
                else:
                    info_weights[information_index][word_index] -= 1/(1/(1 + np.exp((word.vector - action[0].vector))))*step_size

                # use cdist(word.vector, action[0].vector) instead of (word.vector - action[0].vector)
    for energy_index in range(len(energy_vectorlist)):
        lattice = energy_vectorlist[energy_index]
        animal = [] # action choice/personality/behavior metric
        lattice = nlp(lattice)
        # number of words in each word cloud
        # how should we add the weights and initialize them randomly probably
        for word_index in range(len(lattice)):
            word = lattice[word_index]
            labeledAnimalPair = action[1]
            if action[0].has_vector():
                
                # energy animal 
                if labeledAnimalPair[1] == energy_index:
                    energy_weights[energy_index][word_index] += (1/(1 + np.exp((word.vector - action[0].vector))))*step_size
                else:
                    energy_weights[energy_index][word_index] -= 1/(1/(1 + np.exp((word.vector - action[0].vector))))*step_size

print(info_weights)
print("/n")
print(energy_weights)

# This part of the code appears to be trying to update the info_weights and energy_weights lists based on the semantic similarity between the words in the word clouds and a given action word.

# For each action word in the actions list, the code iterates through the lists of word vectors in info_vectorlist and energy_vectorlist. For each list of word vectors, the code then iterates through the individual word vectors and compares them to the action word using the semantic similarity measure provided by the nlp object.

# The code then updates the corresponding weight in the info_weights or energy_weights lists based on the semantic similarity between the word vector and the action word. The size of the update is determined by the step_size variable and a sigmoid function applied to the semantic similarity measure. The specific weight that is updated is determined by the information_index or energy_index variables and the labeledAnimalPair variable, which appears to be a tuple indicating the category (either information or energy) and the subcategory (either B or C for information and either P or S for energy) that the action belongs to.



# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

# # Visualize the info weights
# for information_index in range(len(info_weights)):
#     plt.plot(info_weights[information_index], label=f'Information {information_index}')
# plt.legend()
# plt.show()

# # Visualize the energy weights
# for energy_index in range(len(energy_weights)):
#     plt.plot(energy_weights[energy_index], label=f'Energy {energy_index}')
# plt.legend()
# plt.show()

# # Visualize the word clouds
# for information_index in range(len(info_vectorlist)):
#     wordcloud = WordCloud().generate(info_vectorlist[information_index])
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()

# for energy_index in range(len(energy_vectorlist)):
#     wordcloud = WordCloud().generate(energy_vectorlist[energy_index])
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()


# import matplotlib.pyplot as plt

# # Create a bar chart to visualize the weights for each word in the information word clouds
# plt.bar(range(len(b_word_cloud_1)), info_weights[0])
# plt.xticks(range(len(b_word_cloud_1)), b_word_cloud_1, rotation=90)
# plt.title("Information Word Cloud 1 Weights")
# plt.show()

# plt.bar(range(len(c_word_cloud_2)), info_weights[1])
# plt.xticks(range(len(c_word_cloud_2)), c_word_cloud_2, rotation=90)
# plt.title("Information Word Cloud 2 Weights")
# plt.show()

# # Create a bar chart to visualize the weights for each word in the energy word clouds
# plt.bar(range(len(p_word_cloud_3)), energy_weights[0])
# plt.xticks(range(len(p_word_cloud_3)), p_word_cloud_3, rotation=90)
# plt.title("Energy Word Cloud 3 Weights")
# plt.show()

# plt.bar(range(len(s_word_cloud_4)), energy_weights[1])
# plt.xticks(range(len(s_word_cloud_4)), s_word_cloud_4, rotation=90)
# plt.title("Energy Word Cloud 4 Weights")
# plt.show()


# To use the semantic word vectors from the word clouds to improve the performance of a transformer model for predicting sequences of events representations and generating narratives, you can follow these steps:

# Preprocess the data: Make sure that the data you are using is in a suitable format for training a transformer model. This may involve tokenizing the text and creating sequences of tokens, and possibly padding the sequences to a uniform length.
# Train the transformer model: Use the preprocessed data to train a transformer model using an appropriate training algorithm. This may involve splitting the data into training and validation sets, and using the validation set to evaluate the model's performance during training.
# Incorporate the semantic word vectors: Once you have trained a transformer model, you can modify it to use the semantic word vectors from the word clouds in addition to the token sequences. One way to do this would be to concatenate the word vectors with the token sequences and pass this combined input to the model. Alternatively, you could use the word vectors to modify the model's attention mechanisms, allowing the model to pay more attention to certain words or word clusters when generating narratives.
# Test the modified model: Use a suitable evaluation metric (such as perplexity or BLEU score) to test the performance of the modified model on a test dataset. Compare the results to the performance of the original model to see if the incorporation of the semantic word vectors has improved the model's ability to predict sequences of events representations and generate narratives.
# Visualize the results: You can use various techniques to visualize the performance of the modified model and compare it to the original model. For example, you could plot the evaluation metric scores over time during training, or use techniques like t-SNE to visualize the relationships between the word vectors and the model's output.

# To incorporate the word clouds of semantic vectors into the event representation that the transformer model predicts, you can add the vectors as additional input features to the model. You can do this by concatenating the vectors to the other input features that the model uses to predict the event representation.

# For example, suppose that the transformer model takes in a sequence of words as input and generates a fixed-length vector representation of the sequence as output. To add the semantic vectors, you can concatenate the vectors to the input sequence before it is passed through the model. The transformer will then learn to incorporate the semantic information provided by the vectors when generating the event representation.

# Another option is to use the semantic vectors to augment the event representation produced by the transformer model. For example, you could concatenate the semantic vectors to the event representation vector produced by the model, or you could use the semantic vectors to modulate the event representation vector produced by the model.

# It's worth noting that simply adding the semantic vectors to the input or output of the transformer model may not necessarily improve the quality of the narrative generation. You may need to experiment with different ways of incorporating the semantic vectors and evaluate the performance of the model on a narrative generation task to determine the most effective approach.


# To incorporate the word clouds of semantic vectors into an event2event transformer model, you can try the following approach:

# Preprocess the word clouds by converting them into numeric representations that the transformer can process, such as word embeddings or one-hot encodings.
# Modify the transformer model's architecture to accept the word clouds as additional input alongside the event representation inputs. You may need to add additional input layers or modify the existing input layers to accommodate the word clouds.
# Incorporate the word clouds into the transformer model's training procedure. You can do this by concatenating the word clouds with the event representation inputs and passing them through the transformer model. Alternatively, you can add additional layers to the model that process the word clouds and integrate their output with the event representation inputs.
# Train the modified transformer model on a dataset of event sequences with the word clouds as additional input.
# Evaluate the performance of the modified transformer model on the event sequence prediction task. If the inclusion of the word clouds leads to improved performance, it suggests that the semantic vectors are useful for improving the narrative generation capabilities of the transformer model.
# Note that this is just one possible approach and there may be other ways to incorporate the word clouds into the transformer model. It will depend on the specific details of the model architecture and the training procedure.