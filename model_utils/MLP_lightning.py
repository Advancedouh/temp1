import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os.path as osp
from pytorch_lightning import (
    seed_everything,
)
import pytorch_lightning as pl
from .Tools import SubTaskADataset
from tokenizer_utils import TokenizerFromGlove

class Embedding(nn.Module):
    def __init__(self, embeddings) -> None:
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            torch.Tensor(embeddings)
        )
    
    def forward(self, input_ids, mask=None):
        embeds = self.embeddings(input_ids)
        return {
            'input_embed' : embeds,
            'mask' : mask
        }


class MLPLighting(pl.LightningModule):
    def __init__(self, embed_size, hidden_size, output_size, dropout, lr, batch_size, file_path:dict, dataset_param:dict, tokenizer):
        super().__init__()
        if not hasattr(hidden_size, '__len__'):
            hidden_size = [hidden_size]
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.dataset_param = dataset_param
        self.file_path = file_path #key:[train, val, test]

        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.embed_layer = Embedding(self.tokenizer.vectors) 

        hidden_size_all = [embed_size] + hidden_size + [output_size]
        self.model = nn.ModuleList([
            nn.Linear(hidden_size_all[i], hidden_size_all[i+1]) for i in range(len(hidden_size_all) - 1)
        ])
        
    def pool(self, input_embed, input_mask):
        #input_embed:[batchsize, max_len, embed_size]
        #input_mask:[batchsize, max_len]
        
        pooled_input = (input_embed * input_mask.unsqueeze(-1)).sum(1) / input_mask.sum(-1).unsqueeze(-1)
        return pooled_input

    def forward(self, input_ids, input_mask=None):
        #如果没输入mask，就代表已经池化过
        #如果输入mask，则先进行池化
        embed_and_mask = self.embed_layer(input_ids, input_mask)
        input_embed, input_mask = embed_and_mask['input_embed'], embed_and_mask['mask']
        if input_mask is not None:
            input_embed = self.pool(input_embed, input_mask)
        for fc_layer_except_last in self.model[:-1]:
            input_embed = fc_layer_except_last(input_embed)
            input_embed = self.act(self.dropout(input_embed))
        
        return self.model[-1](input_embed)
    
    def configure_optimizers(self):
        #raise NotImplementedError()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr = self.lr
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        return [optimizer], [lr_scheduler]
    
    def _tokenize(self, sentence):
        #dict{'input_ids', 'mask'}
        #return self.tokenizer(sentence)
        return self.tokenizer(sentence)['input_ids'], self.tokenizer(sentence)['mask']

    def training_step(self, batch, batch_idx):
        #print('training_step:', batch, batch_idx)
        x, y = batch
        input_ids, input_mask = self._tokenize(x)
        y_hat = self.forward(input_ids, input_mask)
        loss = self.criterion(y_hat, y)

        predicts = torch.argmax(y_hat, dim=1)
        
        acc = torch.sum(predicts == y.data) / float(y.shape[0])
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return {
            'loss' : loss,
            'train_acc' : acc
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids, input_mask = self._tokenize(x)
        y_hat = self.forward(input_ids, input_mask)
        loss = self.criterion(y_hat, y)

        predicts = torch.argmax(y_hat, dim=1)
        
        acc = torch.sum(predicts == y.data) / float(y.shape[0])
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return {
            'val_loss' : loss,
            'val_acc' : acc
        }
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        input_ids, input_mask = self._tokenize(x)
        y_hat = self.forward(input_ids, input_mask)
        loss = self.criterion(y_hat, y)

        predicts = torch.argmax(y_hat, dim=1)
        
        acc = torch.sum(predicts == y.data) / float(y.shape[0])
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return {
            'test_loss' : loss,
            'test_acc' : acc
        }
    
    def train_dataloader(self):
        dataset = SubTaskADataset(
            file_path=self.file_path['train'],
            language=self.dataset_param['language'],
            uncase=self.dataset_param['uncase'],
            mark=self.dataset_param['mark'],
            context=self.dataset_param['context']
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        dataset = SubTaskADataset(
            file_path=self.file_path['val'],
            language=self.dataset_param['language'],
            uncase=self.dataset_param['uncase'],
            mark=self.dataset_param['mark'],
            context=self.dataset_param['context']
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        return dataloader
    


