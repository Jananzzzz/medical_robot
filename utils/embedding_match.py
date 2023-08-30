# use faiss-cpu to match embedding
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# get 100 vectors with 768-dim from data.jsonl
embeddings = [] 
with open('data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embedding'])
embeddings = np.array(embeddings).astype('float32')
print("shape of knowledge embeddings: ", embeddings.shape)

query = ["月经失调怎么办？"]
query = ["感冒了怎么办？"]
query = ["头疼怎么办"]
model = SentenceTransformer('/Users/janan/Chinese-medical-dialogue-data/m3e-base')
query_embeddings = model.encode(query)
query_embeddings = np.array(query_embeddings).astype('float32')
# print("shape of query embedding: ", query_embeddings.shape)

# find the 5 most nearest neighbors of query_embeddings in embeddings
index = faiss.IndexFlatL2(768)  # build the index
index.add(embeddings)                  # add vectors to the index
k = 3
D, I = index.search(query_embeddings, k) # sanity check
print("Query: ")
print(query[0])
# fetch the answer of the nearest neighbor
answer_list = []
with open('data.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i in I[0]:
            data = json.loads(line)
            answer_list.append(data['answer'])
            # print(data['answer'])
for i in range(len(answer_list)):
    print("Answer", i+1, ":")
    print(answer_list[i])


# # create 1 million random 256-dim vectors
# d = 768
# nb = 1000000
# np.random.seed(1234)
# xb = np.random.random((nb, d)).astype('float32')
# print("xb: ", xb.shape)

# # create an sample to match
# nq = 1
# xq = np.random.random((nq, d)).astype('float32')
# print("xq: ", xq.shape)

# # find 5 nearest neighbors of xq in xb and print them out
# index = faiss.IndexFlatL2(768)  # build the index
# index.add(embeddings)                  # add vectors to the index
# k = 5
# D, I = index.search(xq, k) # sanity check
# print("Query vector:")
# print(xq)
# print("\nIndices of nearest neighbors:")
# print(I)
# print("\nDistances to nearest neighbors:")
# print(D)

