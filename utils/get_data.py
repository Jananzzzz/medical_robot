import pandas as pd
import os

def get_qa_pair():
    # get data from csv file
    qa_list = []

    data_dir = '/Users/janan/Chinese-medical-dialogue-data/Data_数据/'
    # iterate through all folders under data_dir
    for folder in os.listdir(data_dir):
        # iterate through all csv files under each folder
        for file in os.listdir(os.path.join(data_dir, folder)):
            # check if the file is a csv file
            if file.endswith('.csv'):
                # read data from csv file
                df = pd.read_csv(os.path.join(data_dir, folder, file), encoding='GB18030')
                # print first 10 rows
                # print(df.head(10))
                # print column names
                # for i in df.columns:
                #     print(i, df.head(5)[i].values[0])
                # # print rows count
                # print("#" * 50, "rows count:", len(df))
                # iterate through all rows, add question and answer column to qa_pair
                for index, row in df.iterrows():
                    # fetch 1/10 of the data
                    if index % 10 == 0:
                        qa_pair = []
                        qa_pair.append(row['ask'])
                        qa_pair.append(row['answer'])
                        qa_list.append(qa_pair)
    print("all files read, total qa pairs:", len(qa_list))
    return qa_list

if __name__ == '__main__':
    qa_list = get_qa_pair()
    print(len(qa_list))
    print(qa_list[0])
    question_list = []
    answer_list = []
    for qa_pair in qa_list:
        question_list.append(qa_pair[0])
        answer_list.append(qa_pair[1])

    

