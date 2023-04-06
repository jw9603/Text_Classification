import torch
from torch.utils.data import Dataset


# https://huggingface.co/docs/transformers/main_classes/tokenizer
# https://wikidocs.net/156998
class TextClassificationCollator(): # it is Pytorch DataLoader's collate_fn
    def __init__(self,tokenizer, max_length,with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        
    def __call__(self,samples): # TextClassificationDataset의 return값이 samples로 들어감
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        return_val = {
            'input_ids' : encoding['input_ids'],
            'attention_mask' : encoding['attention_mask'],  #  It has a role to prevent the attention value from entering the padding.
            'labels' : torch.tensor(labels,dtype=torch.long)
        }
        if self.with_text:
            return_val['text'] = texts
        return return_val
        
        
# The Dataset class is a step to construct the entire dataset. 
# As input, you can put the entire x (input feature) and y (label) as tensors.
class TextClassificationDataset(Dataset):
    # This class inherits from Pytorch's Dataset
    # When inheriting from Dataset, only 3 methods are overridden
       # __init__(self) : A method that declares the necessary variables. Load x and y coming as input or load the file list.
       # __len__(self) : x나 y 는 길이를 넘겨주는 메서드.
       # __getitem__(self, index) : Method that returns the "index"th data. It should return tensor
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    # When putting the dataset into the data loader, it creates a mini-batch whenever necessary and returns it at every iteration. -> getitem
    def __getitem__(self,index):
        
        text = str(self.x[index])
        label = self.y[index]
        
        return {
            'text' : text,
            'label' : label
        }