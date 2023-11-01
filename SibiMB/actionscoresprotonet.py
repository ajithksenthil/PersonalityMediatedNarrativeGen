import torch
from torch import nn
from scipy.spatial.distance import cdist
import numpy as np


import transformers
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model tokenizer, you can replace 'bert-base-uncased' with the specific model you want to use
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = AutoModel.from_pretrained('bert-base-uncased')


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

def get_word_vectors(words):
    word_vectors = []
    for word in words:
        # Tokenize the word
        inputs = tokenizer(word, return_tensors="pt")
        # Get the embedding
        outputs = model(**inputs)
        word_embedding = outputs.last_hidden_state[:,0,:].detach().numpy()
        word_vectors.append(word_embedding)
    return np.array(word_vectors)

# Create a list for each word cloud
word_clouds = [B_word_cloud, C_word_cloud, P_word_cloud, S_word_cloud]

# Get the mean of the word embeddings for each word cloud
word_vectors = {}
for i, word_cloud in enumerate(word_clouds):
    word_vectors[i] = np.mean(get_word_vectors(word_cloud), axis=0)

# Create the initial prototypes
init_prototypes = torch.tensor([word_vectors[i] for i in range(4)], dtype=torch.float32)

# Assume word_vectors is a dict with your class labels as keys and the corresponding word vectors as values.
# For example: word_vectors = {'B': np.array([...]), 'C': np.array([...]), 'P': np.array([...]), 'S': np.array([...])}

# init data 
from torch.utils.data import DataLoader, TensorDataset

# Assume X is your input data and y are your labels
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
 

# Create a DataLoader
dataloader = DataLoader(TensorDataset(X, y), batch_size=32)

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=500, output_dim=300):
        super(EmbeddingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ProtoNet(nn.Module):
    def __init__(self, encoder, init_prototypes):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.prototypes = nn.Parameter(init_prototypes, requires_grad=True)

    def forward(self, x):
        z = self.encoder(x)
        dists = self.euclidean_dist(z, self.prototypes)
        return dists

    def euclidean_dist(self, x, y):
        return torch.cdist(x, y)

# Initial prototypes for each class, computed as the mean of the starting word vectors
init_prototypes = torch.tensor([np.mean(word_vectors[label], axis=0) for label in ['B', 'C', 'P', 'S']], dtype=torch.float32)

criterion = nn.CrossEntropyLoss()
model = ProtoNet(EmbeddingNetwork(), init_prototypes)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):  # Number of epochs
    running_loss = 0.0
    for i, data in enumerate(dataloader):  # Assume you're using a DataLoader
        # Get the inputs
        inputs, labels = data
        # Change the type of labels to torch.long
        labels = labels.long()


        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

print('Finished Training')

# Classification of new words
new_words = ['new_word1', 'new_word2', 'new_word3']  # Replace with your new words
new_word_vectors = get_word_vectors(new_words)  # Replace with your function to get word vectors

# Pass the new word vectors through the ProtoNet
dists = model(torch.tensor(new_word_vectors, dtype=torch.float32))

# Get the predicted labels
predicted_labels = dists.argmin(dim=1).numpy()

# Map integer labels back to string labels
label_map = {0: 'B', 1: 'C', 2: 'P', 3: 'S'}
predicted_labels = [label_map[label] for label in predicted_labels]

print(predicted_labels)



# def calculate_prototype_class_scores(actionevents, prototype_net, lattices):
#     for lst in actionevents:
#         for ae in lst:
#             action = ae["predicate"][0]
#             action_vector = ""
#             # action_vector = get_action_vector(action)  # Get the word vector for the action from wherever you're storing them

#             # Get the prototype class scores for BP, CP, CS, and BS
#             prototype_class_scores = prototype_net.forward(action_vector)

#             # Replace the actiontypes scores with the prototype class scores
#             ae["actiontypes"] = prototype_class_scores

#     return actionevents
