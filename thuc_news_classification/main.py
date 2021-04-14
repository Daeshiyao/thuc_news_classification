import paddle
import paddlehub as hub
import os
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset

print('the version of paddle: {}'.format(paddle.__version__))
print('the version of paddlehub: {}'.format(hub.__version__))

DATA_DIR = './thu_news/'

def data_processing(file_name, file_name_new):
    new_file = ''
    with open(DATA_DIR+file_name, 'r', encoding='utf-8') as file_to_read:
        for line in file_to_read:
            new_line = line.split('\t')[1].replace('\n', '') + '\t' +line.split('\t')[0] + '\n'
            new_file += new_line
    file_to_read.close()
    with open(DATA_DIR+file_name_new, 'w', encoding='utf-8') as file_to_write:
        file_to_write.write(new_file)
    file_to_write.close()

def test_data_transfer(test_file):
    data_list = []
    with open(DATA_DIR+test_file, 'r', encoding='utf-8') as file_to_read:
        for text in file_to_read:
            data = text.split('\t')[0]
            data_list.append([data])
    return data_list

class MyDataset(TextClassificationDataset):
    base_path = DATA_DIR
    label_list = ['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']

    def __init__(self, tokenizer, max_seq_len: int = 128, mode: str='train'):
        if mode == 'train':
            data_file = 'train_new.txt'
        elif mode == 'test':
            data_file = 'test_new.txt'
        elif mode == 'valid':
            data_file = 'valid_new.txt'
        else:
            raise ValueError('Unkonwn mode type, choose from train, valid or test!!!')
        super().__init__(
            base_path = self.base_path,
            tokenizer = tokenizer,
            max_seq_len = max_seq_len,
            mode = mode,
            data_file = data_file,
            label_list = self.label_list,
            is_file_with_header=True
        )


#data processing following the data style as 'label \t data \n'
data_processing('train.txt', 'train_new.txt')
data_processing('valid.txt', 'valid_new.txt')
data_processing('test.txt', 'test_new.txt')

#model
model  = hub.Module(name='ernie', task='seq-cls', num_classes=len(MyDataset.label_list))

#get tokenizer
tokenizer = model.get_tokenizer()

#dataset
print('Start to build training dataset...')
train_dataset = MyDataset(tokenizer, mode='train')

print('Start to build validation dataset...')
valid_dataset = MyDataset(tokenizer, mode='valid')

print('Start to build test dataset...')
test_dataset = MyDataset(tokenizer, mode='test')

#print the first and last three data
print_list = [0,1,2,len(valid_dataset)-3,len(valid_dataset)-2,len(valid_dataset)-1]
for idx, data in enumerate(valid_dataset):
    if idx in print_list:
        print(data)
#training
optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ckpt', use_gpu=True)
trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=valid_dataset, save_interval=1)
result = trainer.evaluate(test_dataset, batch_size=32)

#prediction
label_list = MyDataset.label_list
label_mapping = {idx: label_text for idx, label_text in enumerate(label_list)}
data = test_data_transfer('test.txt')

model = hub.Module(name='ernie', task='seq-cls', load_checkpoint='./ckpt/best_model/model.pdparams', label_map=label_mapping)
results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=True)

for idx, text in enumerate(data):
    if idx == 0:
        pass
    else:
        print('Data: {} \t Label: {}'.format(text[0], results[idx]))