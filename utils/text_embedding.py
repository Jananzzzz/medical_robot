from sentence_transformers import SentenceTransformer
from get_data import get_qa_pair
import time
import json
import torch


# model = SentenceTransformer('moka-ai/m3e-base')
device = torch.device('mps')
model = SentenceTransformer('/Users/janan/Chinese-medical-dialogue-data/m3e-base')
model.to(device)
print(device)

example_sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]

qa_list = get_qa_pair()
question_list = []
answer_list = []
for qa_pair in qa_list:
    question_list.append(qa_pair[0])
    answer_list.append(qa_pair[1])

sentences = question_list

print("start embedding")
start = time.time()
embeddings = model.encode(sentences)
end = time.time()
print("total embedding time:", end - start)
print("average embedding time:", (end - start) / len(sentences))
# print(type(embeddings[0])) # <class 'numpy.float32'>

# add (embedding, answer) pair to data.jsonl
with open('data.jsonl', 'a') as f:
    for i in range(len(embeddings)):
        data = {
            "embedding": embeddings[i].tolist(),
            "answer": answer_list[i]
        }
        f.write(json.dumps(data, ensure_ascii=False) + '\n')



#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", type(embedding), embedding.shape)
#     print("")
