import spacy
from spacy import displacy
import re

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    chunks = []
    for sent in doc.sents:
        action = [token.lemma_ for token in sent if (token.dep_ in ("ROOT", "relcl"))]
        if action:
            action = action[0]
            chunk = sent.text
            chunk = re.sub(rf'({action})', r'<b>\1</b>', chunk, flags=re.I)
            chunks.append(chunk)
    return chunks

def to_html(chunks):
    return '<br>'.join(f'<p>{chunk}</p>' for chunk in chunks)

text = "John went to the store. He bought a loaf of bread. Then he returned home."
chunks = process_text(text)
html = to_html(chunks)

print(html)
