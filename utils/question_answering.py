# use faiss-cpu to match embedding
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import tqdm
import torch
import time

# get 100 vectors with 768-dim from data.jsonl
embeddings = [] 
start_time = time.time()
with open('data.jsonl', 'r') as f:
    # load data with dynamic progress bar
    total_lines = sum(1 for line in f)
    f.seek(0)
    for line in tqdm.tqdm(f, total=total_lines, desc="Loading data"):
        data = json.loads(line)
        embeddings.append(data['embedding'])
    # for line in tqdm.tqdm(f):
    #     data = json.loads(line)
    #     embeddings.append(data['embedding'])
embeddings = np.array(embeddings).astype('float32')
end_time = time.time()
print("Finish data loading.", "  time consumed: ", end_time - start_time, "s")

start_time = time.time()
model = SentenceTransformer('/Users/janan/Chinese-medical-dialogue-data/m3e-base')
index = faiss.IndexFlatL2(768)  # build the index
index.add(embeddings)                  # add vectors to the index
k = 10
end_time = time.time()
print("Finish model loading and index building.", "  time consumed: ", end_time - start_time, "s")

while(True):
    print("")
    query = input("请输入您的问题/症状：")
    if query == "exit":
        break
    query_embeddings = model.encode(query)
    query_embeddings = np.array([query_embeddings]).astype('float32')
    D, I = index.search(query_embeddings, k) # sanity check
    # fetch the answer of the nearest neighbor
    answer_list = []
    with open('data.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i in I[0]:
                data = json.loads(line)
                answer_list.append(data['answer'])
                # print(data['answer'])
    answer_list_embeddings = model.encode(answer_list)
    # find the most nearest neighbor of query_embeddings in answer_list_embeddings by cosine similarity
    query_embeddings = torch.from_numpy(query_embeddings)
    answer_list_embeddings = torch.from_numpy(answer_list_embeddings)
    # query_embeddings = query_embeddings.to('cuda')
    # answer_list_embeddings = answer_list_embeddings.to('cuda')
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(query_embeddings, answer_list_embeddings)
    # cos_sim = cos_sim.cpu().detach().numpy()
    # print(cos_sim)
    max_index = np.argmax(cos_sim)
    print("Answer: ", answer_list[max_index])
    # for i in range(len(answer_list)):
    #     print("Answer", i+1, ":")
    #     print(answer_list[i])


