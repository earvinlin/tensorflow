# Listing 6.8 Processing the labels of the raw IMDB data
import os
import platform

if platform.system() == "Windows" :
    imdb_dir = 'C:\\Workspaces\\Datasets\\aclImdb' 
elif platform.system() == "Linux" :
    imdb_dir = '/home/earvin/workspaces/datasets/aclImdb'
else : # Mac path 
    imdb_dir = '/Users/earvin/workspaces/datasets/aclImdb'

train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
#        print('fname= ', fname)
        if fname[-4:] == '.txt':
            # on windows platform, 會因為編碼問題報錯，所以要加上指定編碼
            f = open(os.path.join(dir_name, fname), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                 labels.append(1)
#print("labels counts= ", len(labels))
#print("texts  counts= ", len(texts))

#print("labels[:10]= ", labels[:10])
#print("texts[:1]= ", texts[:1])
#print("texts[1:2]= ", texts[1:2])
#print("texts[2:3]= ", texts[2:3])



# Listing 6.9 Tokenizing the text of the raw IMDB data
from keras.preprocessing.text import Tokenizer # 分詞器；tf v2.15 keras
# 20240130 tf2.12 
#from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
import numpy as np

maxlen = 100               # 100個單詞後截斷評論
training_samples = 200     # 在200個樣本上訓練
validation_samples = 10000 # 在10000個樣本上驗證
max_words = 10000          # 只考慮數據集中前10000個最常見的單詞

tokenizer = Tokenizer(num_words=max_words)
#print("tokenizer= ", type(tokenizer));

tokenizer.fit_on_texts(texts)
print("=== 顯示tokenizer屬性\n")
print("document_count= ", tokenizer.document_count)
print("word_docs= ", tokenizer.word_docs)


sequences = tokenizer.texts_to_sequences(texts)
print("sequences= ", len(sequences))
print("sequence[0]= ", sequences[0])
print("sequence[1]= ", sequences[1])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 將數據劃分為訓練集和驗證集，首先要打亂數據
# 因為一開始數據中的樣本是排序好的
# (所有負面評論在前面，然後是所有正面評論)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]







