from model_utils import MLPLighting
from pytorch_lightning import Trainer
from tokenizer_utils import TokenizerFromGlove
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
file_path = {
    'train' : 'datas\Sem2022_SubTaskA\Data\\train_zero_shot.csv',
    'val' : 'datas\Sem2022_SubTaskA\Data\dev_labeled.csv',
    'test': None
}

dataset_param = {
    'language' : ['EN'],
    'uncase' : True,
    'mark' : True,
    'context' : False
}

tokenizer = TokenizerFromGlove(
    vector_file_path='datas\glove.6B.100d.txt',
    max_length=128,
    extra_token=None,
    return_tensors='pt'
)

model = MLPLighting(
    embed_size=100,
    hidden_size=[25,10],
    output_size=2,
    dropout=0.1,
    lr=1e-1,
    batch_size=8,
    file_path=file_path,
    dataset_param=dataset_param,
    tokenizer=tokenizer
)

trainer = Trainer(
    max_epochs = 5
)

trainer.fit(model)