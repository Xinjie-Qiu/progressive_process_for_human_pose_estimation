import os
import numpy as np




filepath = '/data/lsp_dataset/images/'
list = os.listdir(filepath)
np.random.shuffle(list)
train_set = 'train_set.txt'
eval_set = 'eval_set.txt'
fo_v = open(eval_set, 'w')
fo_t = open(train_set, 'w')
for i, name in enumerate(list):
    if i % 10 == 0:
        fo_v.write(name + '\n')
    else:
        fo_t.write(name + '\n')
fo_v.close()
fo_t.close()
print('yyy')