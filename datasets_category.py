import json
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class PersianDataset(Dataset):
    def __init__(self, f, shuffle,sep_token,input_format):
        '''
        f = path to json file
        sep_token = token separatore(?)
        input_format = perchè i vari json sono organizzati in modo diverso, capire come fare questo
        shuffle = shuffle del dataloader
        '''

        ''' content : list() = all the possible combinations (question, answer)
            labels : list() = list with 0 and 1, 1 where we have the correct answer'''
        

        content, labels = [], []
        x = open(f).readlines()
        if shuffle:
            random.shuffle(x)
        
        self.corr_ans_ids = list()

        for line in x:
            '''create a python dict from this line of the jsonl'''
            instance = json.loads(line)

            ''' possible answers, length may vary from 3 to 5 only one is correct
                for network purposes we  need to have the number of choices fixed
                aribitrary chosen as 4, so we proceed only if len(choices) == 4
            '''
            #choices = [a for a in instance["candidates"]]

            #if len(choices) == 4:
            if len(instance["candidates"]) == 4:
                '''question'''
                question = instance["question"]
                '''id (increasing number from 1 to 4) of the correct answer'''
                correct_answer_id = int(instance["answer"])
                '''create all the possible combinations (question, answer)'''

                c = instance["category"]

                a1 = instance["candidates"][0]
                a2 = instance["candidates"][1]
                a3 = instance["candidates"][2]
                a4 = instance["candidates"][3]
                # for c in choices:
                #     content.append("{} {}".format(question,c))

                if input_format == "0":
                    content.append("Context: {} {} Question: {} {} Answer: {}".format(c, sep_token, question, sep_token, a1))
                    content.append("Context: {} {} Question: {} {} Answer: {}".format(c, sep_token, question, sep_token, a2))
                    content.append("Context: {} {} Question: {} {} Answer: {}".format(c, sep_token, question, sep_token, a3))
                    content.append("Context: {} {} Question: {} {} Answer: {}".format(c, sep_token, question, sep_token, a4))
                elif input_format == "1":
                    content.append("{} \\n {} \\n {}".format(question, a1, c))
                    content.append("{} \\n {} \\n {}".format(question, a2, c))
                    content.append("{} \\n {} \\n {}".format(question, a3, c))
                    content.append("{} \\n {} \\n {}".format(question, a4, c))
                elif input_format == "2":
                    content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(c, question, a1))
                    content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(c, question, a2))
                    content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(c, question, a3))
                    content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(c, question, a4))
                elif input_format == "3":
                    content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a1, c, sep_token))
                    content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a2, c, sep_token))
                    content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a3, c, sep_token))
                    content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a4, c, sep_token))
                
                # content.append("{} {}".format(question, a1))
                # content.append("{} {}".format(question, a2))
                # content.append("{} {}".format(question, a3))
                # content.append("{} {}".format(question, a4))

                if correct_answer_id == 1:
                    labels += [1, 0, 0, 0]
                elif correct_answer_id == 2:
                    labels += [0, 1, 0, 0]
                elif correct_answer_id == 3:
                    labels += [0, 0, 1, 0]
                elif correct_answer_id == 4:
                    labels += [0, 0, 0, 1]

                # '''1, 2, 3, 4 '''
                # answers = [1,2,3,4]
                # y = [0,0,0,0]
                # y[answers.index(correct_answer_id)] = 1    #crea una lista concatenata di 0 e 1, 1  alla posizione della risposta corretta
                # labels += y
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self,index):
        s1,s2 = self.content[index], self.labels[index]
        return s1,s2

    def get_labels_gt(self):
      return self.corr_ans_ids
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
            
