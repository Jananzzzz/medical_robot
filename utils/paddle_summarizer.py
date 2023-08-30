from paddlenlp import Taskflow

# this summarization is based on Randeng-Pegasus-523M-Summary-Chinese-SSTIA

summarizer = Taskflow("text_summarization")
print(summarizer('2022年，中国房地产进入转型阵痛期，传统“高杠杆、快周转”的模式难以为继，万科甚至直接喊话，中国房地产进入“黑铁时代”'))

