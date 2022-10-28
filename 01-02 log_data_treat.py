 #%%
# default setting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime, timedelta

os.chdir(r'N:\[03] 단기 작업\빅콘 테스트')


#%%
log_data = pd.read_csv('log_data.csv')

# chk var type
log_data.info()

# change datetime
log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])

# chk var type
log_data.info()

# sort data by ['user_id', 'timestamp']
log_data = log_data.sort_values(['user_id', 'timestamp'])

#%%
# NA Filling 
# check_na
print(log_data.isna().sum())

# 결과
#user_id                0
#event                  0
#timestamp              0
#mp_os                980
#mp_app_version    660597
#date_cd                0


# 최빈값 찾기
log_data.mp_os.mode() # Android
log_data.mp_app_version.mode() # 3.14.0

#%%

# 앞에서 메우기
ndf = []
for idx, df in tqdm(log_data.groupby('user_id')):
    ndf.append(df.fillna(method='ffill'))

log_data = pd.concat(ndf, axis = 0, ignore_index=True)

# 개수 체크
print(log_data.isna().sum())

# 결과
#user_id               0
#event                 0
#timestamp             0
#mp_os                50
#mp_app_version    83973
#date_cd               0

#%%

# 뒤에서 메우기
counter = 0; ndf = []
for idx, df in tqdm(log_data.groupby('user_id')):
    ndf.append(df.fillna(method='bfill'))

log_data = pd.concat(ndf, axis = 0, ignore_index=True)

# 개수 체크
print(log_data.isna().sum())

# 결과 -> 이걸로는 해결안됨. 순수하게 결측으로만 있는 값들이 이정도 됨.
#user_id               0
#event                 0
#timestamp             0
#mp_os                14
#mp_app_version    32571
#date_cd               0
#dtype: int64

#%%

# 둘 다 결측인 애들.
# 최빈값 메우기
counter = 0; ndf = []
for idx, df in tqdm(log_data.groupby('user_id')):
    line = df
    line.mp_os = line.mp_os.fillna('Android')
    line.mp_app_version = line.mp_app_version.fillna('unknown')
    
    ndf.append(line)
    
log_data = pd.concat(ndf, axis = 0, ignore_index=True)

# 개수 체크
print(log_data.isna().sum())

# 결과
# user_id           0
# event             0
# timestamp         0
# mp_os             0
# mp_app_version    0
# date_cd           0

# 파일 저장 -> 이이후는 전처리 작업
log_data.to_csv('NA_fill_log_data.csv', index=False)

#%%

log_data = pd.read_csv('NA_fill_log_data.csv')
usr_data = pd.read_csv('user_spec.csv')

#%%

# 분포 개수만 쓰고 싶으면 이렇게 쓰면 되지.
log_data['counter'] = 1

# 이렇게 하면 전처리 끝.
event_count = log_data.pivot_table(index = 'user_id', columns='event', values = 'counter', aggfunc = 'sum').reset_index()

event_count.to_csv('event_count.csv', index=False)

#%%

user_log = []

for idx, val in tqdm(log_data.groupby('user_id')):
    event = [i for i in val.event.values]
    mp_os = [i for i in val.mp_os.values]
    mp_app_version = [i for i in val.mp_app_version.values]
    timestamp = [i for i in val.timestamp.values]
    
    user_log.append(pd.DataFrame({'event': [event], 'mp_os' : [mp_os], 'mp_app_version' : [mp_app_version], 'timestamp' : [timestamp]}, index = [idx]))
  
user_log = pd.concat(user_log, axis = 0)
user_log = user_log.rename_axis('user_id').reset_index()  


# 이 작업을 user_spec에 시간 변수에 따라서, 연산해야한다.
user_log['latest_os'] = user_log['mp_os']
user_log['latest_version'] = user_log['mp_app_version']
user_log['update_times'] = user_log['mp_app_version']

user_log['latest_os'] = user_log['latest_os'].apply(lambda x: x[-1])
user_log['latest_version'] = user_log['latest_version'].apply(lambda x: x[-1])
user_log['update_times'] = user_log['update_times'].apply(lambda x: len(set(x)))

print('1차 작업 종료. 2차작업 시작->4시간 이상 소요')

#%%
# 그러면 지금까지 발생횟수랑, userlog형을 꺼내서 연산해야할것.


event_col = log_data.event.unique()
usr_log_plus = []
null_counter = 0

for i in tqdm(usr_data.index):
    line = usr_data.loc[i]
    finder = user_log[user_log.user_id == line.user_id]
    if finder.empty:
        null_counter += 1
        print(f'\nDataFrame is empty!, null set num : {null_counter}', end = ' ')
        usr_log_plus.append(pd.concat([pd.DataFrame({'application_id' : [line.application_id], 'user_id' : ['no log'], 'event': ['no log'], 'mp_os' : ['no log'], 'mp_app_version' : ['no log'],
                                 'timestamp' : ['no log'], 'latest_os' : ['no log'], 'latest_version' : ['no log'],
                                 'update_times' : ['no log']}, index = [0]),
                   pd.DataFrame(np.array(['no log' for i in range(len(event_col))]).reshape(1,-1), columns = event_col, index = [0])], axis = 1))
        continue
    time_idx = (pd.to_datetime(line.insert_time) - timedelta(weeks=4) <= pd.Series(finder.timestamp.iloc[0], dtype = 'datetime64[ns]')) & (pd.Series(finder.timestamp.iloc[0], dtype = 'datetime64[ns]') <= line.insert_time)
    
    event = pd.Series(finder.event.iloc[0])[time_idx].tolist()
    mp_os = pd.Series(finder.mp_os.iloc[0])[time_idx].tolist()
    mp_app_version = pd.Series(finder.mp_app_version.iloc[0])[time_idx].tolist()
    
    try:
        latest_os = mp_os[-1]
    except:
        latest_os = 'unknown'
        
    try:
        latest_version = mp_app_version[-1]
    except:
        latest_version = 'unknown'
    
    temp = []
    for i in event_col:
        temp.append(event.count(i))
    
    usr_log_plus.append(pd.concat([pd.DataFrame({'application_id' : [line.application_id], 'user_id' : [line.user_id], 'event': [event], 'mp_os' : [mp_os], 'mp_app_version' : [mp_app_version],
                             'timestamp' : [timestamp], 'latest_os' : [latest_os], 'latest_version' : [latest_version],
                             'update_times' : [len(set(mp_app_version))]}, index = [0]),
               pd.DataFrame(np.array(temp).reshape(1,-1), columns = event_col, index = [0])], axis = 1))

usr_log_plus = pd.concat(usr_log_plus, axis = 0).reset_index(drop=True)

usr_log_plus.to_csv('applicationid_append_user_spec_restored.csv', index=False)

#%%

usr_log_plus = pd.read_csv('applicationid_append_user_spec_restored.csv')



#%%
'''
이 방안은 컷. 그냥 위 변수들을 user_spec에 붙일거니까...

#%%

line = user_log.copy()

# Modelling
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import logging
from scipy import stats
from time import time


print('구매 로그 1인 사람, 훈련 불가 딕셔너리 제거. word to vec 전처리') 
# minimum len = 2 
short_data_idx = []
for i in tqdm(range(line.shape[0]), leave=True):
    if len(line.iloc[i].event) <= 1:
        short_data_idx.append(line.iloc[i].user_id)

# preprocessing dataset
line = line[~line['user_id'].isin(short_data_idx)].reset_index(drop = True)

# 제거된 유저수
print(f'제거된 유저는 {len(short_data_idx)}명 입니다.')

#%%
# word2vec callback
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss
        
#%%

# train_test_split = book log Word2Vec
clean_log = line.event

# train word2vec model
# 훈련에 들어갈 로그가 유저별로 상당히 부족한편, window는 5로 설정하였음. -> 11개 단어니 20차원으로 적게 실행함.
model = Word2Vec(window = 5, vector_size=5, sg = 1, hs = 0, negative = 20, alpha=0.03, min_alpha=0.0007)
logging.disable(logging.NOTSET) # enable logging
t = time()
model.build_vocab(clean_log)
logging.disable(logging.INFO) # disable logging
callback = Callback() # instead, print out loss for each epoch
t = time()

# Word2vec train train_set - epoch 100 << check -> 50으로 일단 저장해서 놔뒀다. 100으로 다시 돌릴것.
model.train(clean_log,
            total_examples = model.corpus_count,
            epochs = 70,
            compute_loss = True,
            callbacks = [callback]) 

model.save("log2vec.model")

# 시간 로그 제거.
del t

#%%
# OR load model
# 모델 훈련과정 스킵가능. 
model = Word2Vec.load("log2vec.model")

print(model)

# models vector(weight)
X = model.wv.get_normed_vectors() 
print(X)

# models vector(weight) shape
print(X.shape)

# print vector (row = book, col = dim 100 embedding)
plt.figure(figsize=(20, 10)) 
sns.heatmap(X[0:11], cbar = False, cmap='PuBu')
plt.title("11 event's embedding vector visualize",fontsize=12)
plt.show()

#%%
# 유저별 벡터로 임베딩하기
# 우리는 Word2Vec을 상품 단위로 보고 상품 추천으로도 볼 수 있지만
# 유저별로 Word2Vec을 실행한 결과값을 받을수도 있을것이다.
# 따라서, 이 결과를 구매 이력-클릭 이력으로 넣으면 어떨까 하고 생각을 해봄.

# 당연히 이 결과는 tSNE를 쓰든, PCA를 쓰든 축소된 값이 군집화 일어나야할것.
# 일단 데이터셋 부터 만들면서 생각해보자.

# 이제 길이가 다른 구매내역을 100차원에 임베딩을 시킬수 있게 되었다.
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in tqdm(list_of_docs):
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.sum(axis=0) # 원래는 mean 이지만, 이번 경우는 횟수 영향이 중요, 단어가 별로 안중요했다.
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


#%% 임베딩 값 반환.
tokenized_docs = list(user_log['event'])
vectorized_docs = vectorize(tokenized_docs, model=model)
len(vectorized_docs), len(vectorized_docs[0])

# 우리는 이제 유저의 구매 내역을 임베딩한 결과를 얻게 되었다.
# 중요한 사실은 이제 클릭, 구매를 1번 이하로 한 사람은 군집화 할 필요성을 못느꼈다는 가정이 필요하다.

vectorized_docs[:5]

# 이 변수를 유저별로 추가해주면 해결될것.
w2v_df = pd.concat([user_log.user_id.to_frame(), pd.DataFrame(vectorized_docs, columns = ['w2v_'+str(i) for i in range(1,6)])], axis = 1)
w2v_df.to_csv('userlog_w2v_df.csv', index=False)

#%%

w2v_df = pd.read_csv('w2v_df.csv')
'''