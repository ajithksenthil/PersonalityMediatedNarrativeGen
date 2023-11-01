def create_table(conn):

    create_word_vectors_table = """CREATE TABLE IF NOT EXISTS word_vectors (
                                       id INTEGER PRIMARY KEY,
                                       word TEXT NOT NULL UNIQUE,
                                       vector BLOB NOT NULL,
                                       weight REAL
                                   );"""

    try:
        cursor = conn.cursor()
        cursor.execute(create_word_vectors_table)
    except Error as e:
        print(e)


def add_word_vector(conn, word_vector):
    sql = '''INSERT OR IGNORE INTO word_vectors(word, vector, weight)
             VALUES(?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, word_vector)
    conn.commit()
    return cur.lastrowid

def update_word_vector(conn, word, new_vector, new_weight=None):
    sql = '''UPDATE word_vectors
             SET vector = ?,
                 weight = ?
             WHERE word = ?'''
    cur = conn.cursor()
    cur.execute(sql, (new_vector, new_weight, word))
    conn.commit()

def get_word_vector(conn, word, with_weight=False):
    cur = conn.cursor()
    cur.execute("SELECT word, vector, weight FROM word_vectors WHERE word=?", (word,))

    result = cur.fetchone()
    if result and not with_weight:
        result = (result[0], result[1])

    return result



# Path: Databases/wordvecdatabase.py

import numpy as np

# Adding word vectors
word_vector_1 = ("word1", np.array([1.0, 2.0, 3.0]).tobytes(), 0.5)
word_vector_2 = ("word2", np.array([4.0, 5.0, 6.0]).tobytes(), None)
add_word_vector(conn, word_vector_1)
add_word_vector(conn, word_vector_2)

# Updating word vectors
new_vector = np.array([7.0, 8.0, 9.0]).tobytes()
new_weight = 0.7
update_word_vector(conn, "word2", new_vector, new_weight)

# Retrieving word vectors
result = get_word_vector(conn, "word1", with_weight=True)
word, vector_bytes, weight = result
vector = np.frombuffer(vector_bytes, dtype=np.float64)
print(f"{word}: vector={vector}, weight={weight}")

result = get_word_vector(conn, "word2")
word, vector_bytes = result
vector = np.frombuffer(vector_bytes, dtype=np.float64)
print(f"{word}: vector={vector}")


# ------------------------------

import numpy as np
from gensim.models import KeyedVectors

# Load the Google News word2vec model
model_path = 'GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Example lists of words
words_list_1 = ["king", "queen"]
words_list_2 = ["apple", "orange"]
words_list_3 = ["car", "bus"]
words_list_4 = ["dog", "cat"]

all_word_lists = [words_list_1, words_list_2, words_list_3, words_list_4]

# Iterate through the lists and add word vectors to the database
for word_list in all_word_lists:
    for word in word_list:
        if word in word2vec_model:
            word_vector = word2vec_model[word]
            vector_bytes = np.array(word_vector).tobytes()
            add_word_vector(conn, (word, vector_bytes, None))
        else:
            print(f"The word '{word}' is not in the pre-trained word2vec model.")
