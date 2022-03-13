import torch
from torch.utils.data import Dataset
import csv
import re
HEADER_TRAIN = ['DataID','Language','MWE','Setting','Previous','Target','Next','Label']
HEADER_DEV = ['ID','Language','MWE','Previous','Target','Next','Label']
HEADER_TARGET = ['Sentence', 'Label']

class SubTaskADataset(Dataset):
    def __init__(self, file_path, language=['EN', 'PT'], uncase=True, mark=True, context=False) -> None:
        super().__init__()
        self.all_fields = ['ID','Language','MWE','Previous','Target','Next','Label']
        self.language = language
        self.uncase = uncase
        self.mark = mark #标注MWE
        self.context = context #加入上下文

        self._load_data(file_path)

    def _load_data(self, file_path):
        header, datas = None, []
        with open(file_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if header is None:
                    header = row
                    continue
                datas.append(row)
        
        X, y = [], []
        for data in datas:
            if data[header.index('Language')] not in self.language:
                continue
                   
            pre, next = data[header.index('Previous')], data[header.index('Next')]
            target = data[header.index('Target')]    

            if self.uncase is True:
                pre, target, next = pre.lower(), target.lower(), next.lower()

            if self.mark:
                mwe = data[header.index('MWE')]
                pattern = re.compile(rf'{mwe}', re.I)
                repl = f' [SEP] {mwe} [SEP] '
                target = re.sub(pattern, repl, target)

            if self.context:
                sentence = '. '.join([
                    pre, target, next
                ])
            else:
                sentence = target
            
            label = int(data[header.index('Label')])

            X.append(sentence)
            y.append(label)
        
        self.X, self.y = X, y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) :
        return self.X[index], self.y[index]