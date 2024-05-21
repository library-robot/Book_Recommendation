import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import tensorflow as tf 
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from keras.callbacks import ModelCheckpoint
import random


Library = pd.read_csv('C:/Users/Master/.spyder-py3/book_data_fin_3.csv', encoding='cp949') #전체 책 데이터

#Library = Library[['title','isbn','genre']] 
Library_book_borrow = pd.read_csv('C:/Users/Master/.spyder-py3/회원대출0411.csv') #전체 대출 내역
book_borrow_user_sort = Library_book_borrow.sort_values(by = ['member_id']) #사용자이름으로 정렬
book_borrow_user_birth_sort = book_borrow_user_sort.sort_values(by = ['birthday']) #생일로 정렬 
book_borrow_user_birth_sort['year'] = book_borrow_user_birth_sort['birthday'].str.split('-').str[0].astype(int) #년도 추출 (나이 구분)
book_borrow_user_birth_year = book_borrow_user_birth_sort.sort_values(by='year') #년도 추가 



#연령대별 구분
corrent_year = datetime.now().year

over_70s = book_borrow_user_birth_year[book_borrow_user_birth_year['year'] < corrent_year - 70] 
sixties = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 70) & (book_borrow_user_birth_year['year'] < corrent_year - 60)]
fifties = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 60) & (book_borrow_user_birth_year['year'] < corrent_year - 50)]
forties = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 50) & (book_borrow_user_birth_year['year'] < corrent_year - 40)]
thirties = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 40) & (book_borrow_user_birth_year['year'] < corrent_year - 30)]
twenties = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 30) & (book_borrow_user_birth_year['year'] < corrent_year - 20)]
teens = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 20) & (book_borrow_user_birth_year['year'] < corrent_year - 10)]
under_teens = book_borrow_user_birth_year[(book_borrow_user_birth_year['year'] >= corrent_year - 10 )]

#인덱스 정렬 

over_70s.reset_index(drop = False, inplace = True)
sixties.reset_index(drop = False, inplace = True)
fifties.reset_index(drop = False, inplace = True)
forties.reset_index(drop = False, inplace = True)
thirties.reset_index(drop = False, inplace = True)
twenties.reset_index(drop = False, inplace = True)
teens.reset_index(drop = False, inplace = True)
under_teens.reset_index(drop = False, inplace = True)

#성별 구분 
female = book_borrow_user_birth_year[(book_borrow_user_birth_year['gender'] == 'F')]
male = book_borrow_user_birth_year[(book_borrow_user_birth_year['gender'] == 'M')]


#인덱스 정렬 

female.reset_index(drop = False, inplace = True)
male.reset_index(drop = False, inplace = True)


#카테고리 구분 

natural_science = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '자연과학')]
literature  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '문학')]
other  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '총류')]
social_science  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '사회과학')]
descriptive_science  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '기술과학')]
religion  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '종교')]
history  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '역사')]
philosophy  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '철학')]
art  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '예술')]
language  = book_borrow_user_birth_year[(book_borrow_user_birth_year['genre'] == '언어')]

#인덱스 정렬 

natural_science.reset_index(drop = False, inplace = True)
literature.reset_index(drop = False, inplace = True)
other.reset_index(drop = False, inplace = True)
social_science.reset_index(drop = False, inplace = True)
descriptive_science.reset_index(drop = False, inplace = True)
religion.reset_index(drop = False, inplace = True)
history.reset_index(drop = False, inplace = True)
philosophy.reset_index(drop = False, inplace = True)
art.reset_index(drop = False, inplace = True)
language.reset_index(drop = False, inplace = True)

                        
#대출 구분 

borrow_T = book_borrow_user_birth_year[(book_borrow_user_birth_year['lend_status'] == 'T')]    #대출 불가     
borrow_F = book_borrow_user_birth_year[(book_borrow_user_birth_year['lend_status'] == 'F')]    #대툴 가능 

#인덱스 정렬 

borrow_T.reset_index(drop = False, inplace = True)
borrow_F.reset_index(drop = False, inplace = True)


def categories(file) :
    categories = np.zeros((len(file),10))
    for n in range(len(file)) : #카테고리 데이터에 대한 One-Hot Encoding
        if file[n] == '총류' :
            categories[n,0] = 1
        elif file[n] == '자연과학' :
            categories[n,1] = 1
        elif file[n] == '사회과학' :
            categories[n,2] = 1
        elif file[n] == '기술과학' :
            categories[n,3] = 1
        elif file[n] == '종교' :
            categories[n,4] = 1
        elif file[n] == '역사' :
            categories[n,5] = 1
        elif file[n] == '철학' :
            categories[n,6] = 1
        elif file[n] == '문학' :
            categories[n,7] = 1
        elif file[n] == '예술' :
            categories[n,8] = 1
        elif file[n] == '언어' :
            categories[n,9] = 1

    return categories   

def gender(file) :
    gender = np.zeros((len(file),3))
    for n in range(len(file)) : #성별 데이터에 대한 One-Hot Encoding
        if file[n] == 'M' :
            gender[n,0] = 1
        elif file[n] == 'F' :
            gender[n,1] = 1
        else : #성별정보 없음
            gender[n,2] = 1
    return gender   

def borrow(file) :
    borrow = np.zeros((len(file),2))
    for n in range(len(file)) : #대출 데이터에 대한 One-Hot Encoding
        if file[n] == 'F' :
            borrow[n,0] = 1
        elif file[n] == 'T' :
            borrow[n,1] = 1
    return borrow  

def age(file) :
    age = np.zeros((len(file),9))
    corrent_year = datetime.now().year
    for n in range (len(file)) : #나이 데이터에 대한 One-Hot Encoding
        age_s = corrent_year - file[n] #나이 계산    
        if (0 < age_s <= 9) : # 0 ~ 9세
            age[n,0] = 1
        elif (9 < age_s <= 19): # 10 ~ 19세
            age[n,1] = 1
        elif (19 < age_s <= 29): # 20 ~ 29세
            age[n,2] = 1
        elif (29 < age_s <= 39): # 30 ~ 39세
            age[n,3] = 1
        elif (39 < age_s <= 49): # 40 ~ 49세
            age[n,4] = 1
        elif (49 < age_s <= 59): # 50 ~ 59세
            age[n,5] = 1
        elif (59 < age_s <= 69): # 60 ~ 69세
            age[n,6] = 1
        elif (69 < age_s ): # 70세 이상
            age[n,7] = 1
        else : #나이 정보 없음
            age[n,8] = 1

    return age   

   
delete = book_borrow_user_birth_year.drop_duplicates('isbn')
delete.reset_index(drop = False, inplace = True)

duplicates = pd.merge(Library, delete['isbn'], on ='isbn')

# 중복된 행을 제거합니다.
#result = Library[~Library['isbn'].isin(duplicates['isbn'])].dropna(axis = 'index', how = 'any', subset =['isbn'])

def delete_(data) :
    result = Library[~Library['isbn'].isin(data['isbn'])].dropna(axis = 'index', how = 'any', subset =['isbn'])
    return result 

result = delete_(duplicates)
result.reset_index(drop = False, inplace = True)


one_hot = categories(result['genre'])

Library_book_gender_year_add = pd.merge(Library, book_borrow_user_birth_year[['isbn','gender','year']], on ='isbn',how = 'left')
Library_book_gender_year_add.reset_index(drop = False, inplace = True)

Library_book_borrow_add = pd.merge(Library, book_borrow_user_birth_year[['isbn','lend_status']], on ='isbn',how = 'left')
Library_book_borrow_add['lend_status']  = Library_book_borrow_add['lend_status'].fillna('F')
Library_book_borrow_add = Library_book_borrow_add.drop_duplicates('isbn')
Library_book_borrow_add.reset_index(drop = False, inplace = True)



grouped_dict = dict(tuple(book_borrow_user_birth_year.groupby('member_id')))
keys = [group for group in grouped_dict]
#print(grouped_dict[keys[3]])
for i in range(len(keys)) :
    grouped_dict[keys[i]].reset_index(drop = False, inplace = True)








aqwer1 = np.vstack(categories(Library_book_gender_year_add['genre'])),age(Library_book_gender_year_add['year'])

max_len = max(len(seq) for seq in aqwer1)

padded_sequences = [pad_sequences(seq, maxlen=9, padding='post', truncating='post') for seq in aqwer1]

reshape1 = np.expand_dims(padded_sequences[0], axis = 1)
reshape2 = np.expand_dims(padded_sequences[1], axis = 1)
x_train = np.concatenate((reshape1,reshape2),axis = 1)
y_train = gender(Library_book_gender_year_add['gender'])

us1 = 8
user_info = grouped_dict[keys[us1]]
user_data_test = gender(user_info['gender'])



'''
user_info_test = gender(user_info['gender'])
max_len_user = max(len(seq1) for seq1 in user_info_test)
user_data_test = gender(user_info['gender'])
'''


'''
pad_sequences_user = [pad_sequences(seq1, maxlen=3, padding='post', truncating='post') for seq1 in user_info_test]

user_reshape1 = np.expand_dims(pad_sequences_user[0], axis = 1)
user_reshape2 = np.expand_dims(pad_sequences_user[1], axis = 1)
user_data = np.concatenate((user_reshape1[0],user_reshape2[0]),axis = 1)
user_data_test1 = user_data.reshape(2,9)
user_data_test = user_data_test1.astype(np.float64)
'''
'''

model = keras.models.Sequential()
model.add(keras.layers.SimpleRNN(256,activation='tanh',input_shape=(2,9)))
model.add(keras.layers.Dense(128, activation='tanh'))
model.add(keras.layers.Dense(32, activation='tanh'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs_ = 10
with tf.device("/device:GPU:0"):
    history = model.fit(x_train,y_train, epochs=epochs_,batch_size=400)
    optimizer = keras.optimizers.Adam(learning_rate=0.00001)
#model.save('sw_lib_model1.h6')  


'''




new_model1 = keras.models.load_model('sw_lib_model1.h6')

tt =  new_model1.predict(x_train)

ref_matrices = np.tile(user_data_test, (max_len, 1, 1))

# 각 행에서 최대값의 인덱스 찾기
max_indices = np.argmax(ref_matrices, axis=2)


# 결과 행렬 생성
result_matrix = np.zeros_like(tt)

# 각 행에서 최대값의 인덱스에 해당하는 위치에만 값을 설정
for i in range(ref_matrices.shape[0]):
    for j in range(ref_matrices.shape[1]):
        max_index = max_indices[i, j]
        result_matrix[i, max_index] = tt[i, max_index]

ref_matrices = np.tile(user_data_test, (max_len, 1, 1))
similarities_per_matrix = []
asd = []
'''
for target_matrix in result_matrix:
    similarity = cosine_similarity(user_data_test[0].reshape(1,-1), target_matrix.reshape(1, -1))
    asd.append(similarity)
    similarities_per_matrix.append(similarity[0][0])
  '''  
#코사인 유사도
for target_matrix in tt:
    similarity = cosine_similarity(user_data_test[0].reshape(1,-1), target_matrix.reshape(1, -1))
    asd.append(similarity)
    similarities_per_matrix.append(similarity[0][0])


#유사도를 DF에 추가
similarities_per_matrix = np.array(similarities_per_matrix)
recommendations = Library_book_gender_year_add.copy()
recommendations['similarity'] = similarities_per_matrix

#isbn을 기준으로 중복된 책 제거 
delete1 = recommendations.drop_duplicates('isbn')

#recommendations = recommendations.sort_values(by='similarity', ascending=False)

#유저의 대출기록에서 중복된 정보 제거 
delete2 = book_borrow_user_birth_year.drop_duplicates('isbn')

#isbn을 기준으로 현재 대출 상태 정보를 추가 
delete1 = pd.merge(delete1, delete2[['isbn','lend_status']], on ='isbn',how = 'left')

#유저의 대출기록에서 상태를 가져옴으로 대출기록이 존재하지 않는 경우 lend_state를 부여 
delete1['lend_status']  = delete1['lend_status'].fillna('F')


result = delete1[~delete1['isbn'].isin(user_info['isbn'])].dropna(axis = 'index', how = 'any', subset =['isbn'])

#result['lend_status'] = result['lend_status'].replace('T')

sorted_result = result.sort_values(by='similarity', ascending = False)




max_value = sorted_result.iloc[0]['similarity']
max_columns = sorted_result[sorted_result['similarity'] == max_value]       

if(len(max_columns) < 3) :
    random_column = max_columns
else :                
    random_column = max_columns.sample(3)         
  
       
print(random_column)








