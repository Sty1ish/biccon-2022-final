#%%
# default setting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

os.chdir(r'N:\[03] 단기 작업\빅콘 테스트')

rst_data = pd.read_csv('loan_result.csv')
log_data = pd.read_csv('log_data.csv')
usr_data = pd.read_csv('user_spec.csv')
usr_log_plus = pd.read_csv('applicationid_append_user_spec_restored.csv')

#%%
# EDA 특이사항. 이건 뭐지... 대출금리, 금액의 결측은 항상 1이거나 결측일때만 결측이 났다.
print(rst_data[rst_data.loan_rate.isna()].is_applied.value_counts())
print(rst_data[rst_data.loan_rate.isna()].is_applied.info())

pp = rst_data[rst_data.loan_rate.isna()]
aa = []
for i in pp.index:
    aa.append((pp.loc[i].bank_id, pp.loc[i].product_id))

aa = set(aa)
pp.loc[:,['bank_id','product_id']].bank_id

#%%
# user_spec의 전처리 실시.

# yearly income은 6개 그냥 제거하기 위해. 지워버림.
# usr_data = usr_data.dropna(subset=['yearly_income']) # yearly_income 결측값인 행 제거 

# company enter month는 년 월까지로만 변형할 필요가 있다고 봄. (근무 개월. 파생변수화 할것인지) -> 년월일 포맷과 년 월 포맷이 섞였다.


# personal_rehabilitation_yn => na를 0으로 치환

# personal_rehabilitation_complete_yn는 personal_rehabilitation_yn가 0인데 0인값은 -1 치환, na는 -1

# usr_data에서는 결측값 앞으로 밀고 뒤로 밀고 한번 하고 작업실시. -> 무의미한 결측이 많았다. 한명의 유저가 동일 날짜 다른 시간대 다른 기록들....


# existing_loan_cnt / existing_loan_amt는 nan nan 은 0,0으로  채우고
# 1, nan일때는 minian으로 채우기

# loan limit는 희망금액으로 채워버리고
# loan rate는 등장한 최저값으로 채워버리자. (aplication id)


# 전처리 완료된 데이터를 사용함. (코드 준희거 참조.)
# usr_data = pd.read_csv('user_spec_fillna.csv')
usr_data = pd.read_csv('user_spec_filled_final.csv') # 이걸로 파일 변경 마지막에 준 knn 완료파일.
usr_data.isna().sum()

#%%
# 단순 EDA

log_data.event.value_counts().plot(kind='bar')
log_data.mp_app_version.value_counts().sort_index().plot(kind='bar')
log_data.date_cd.value_counts().sort_index().plot()


# 대충 한명당 로그의 수. -> 극단이 꽤 큰편.
log_data.groupby('user_id').event.agg(['count']).hist(bins=100)
log_data.groupby('user_id').event.agg(['count']).plot.box()
log_data.groupby('user_id').event.agg(['count']).describe()

# 발견한 생각보다 큰 문제
chk_df = pd.merge(rst_data, usr_data, how = 'left', on = 'application_id')

print(chk_df.isna().sum()) # rst_data에 결합한 usr_data기준으로 없는 유저가 113명이나 존재함. 훈련샘플에서 예외로 갖다 버리던...

# 유저 정보에서 na가 있는 df >> 아래 데이터 작업이 전체 다 무의미해짐 >> 천만 다행이게도 6월에는 이딴 예제가 없다. 그냥 갖다 던져버리자.
na_df = chk_df[chk_df.user_id.isna()]

# rst_data에서 유저정보가 결측인 인덱스를 제거
rst_data = rst_data.drop(na_df.index)

# 이정도 남았다.
rst_data.isna().sum()

#%%
# 다시 merge한 결과에서 체크를 다시 실시한다.
chk_df = pd.merge(rst_data, usr_data, how = 'left', on = 'application_id')
print(chk_df.isna().sum()) 

#%%
# 현재 존재하는 결측의 수. 4개의 변수에서 결측이 등장.
print(usr_data.isna().sum()) # KNN 완료하여 받았기에 이 KNN 구문은 바로 패스함.
print(usr_data.info())
'''
# knn-imputer 진행
from sklearn.impute import KNNImputer


# 라벨 인코딩 진행. - 전체 열 이름.
col_names = usr_data.columns

# object형이 존재한다. - 라벨 인코딩 이상 필요. (onehot은 sparse mtx니까 거부.)
# personal_rehabilitation 경우는 0~2로 코딩되었으니 변경할 필요는 없고.
label_col = usr_data.select_dtypes('object').columns

# KNN imputer에는 application_id만큼은 군집화에서 빠졌으면 좋겠음.
app_id_data = usr_data.pop('application_id')

# 라벨 인코딩 실시.
label_dict = {}

for i in label_col:
    label_dict[i] = {val : idx for idx, val in enumerate(usr_data[i].unique())}
    usr_data[i] = usr_data[i].map(label_dict[i])


#임퓨터 선언(가장 가까운 값으로 대체하겠다.)
imputer=KNNImputer(n_neighbors=1)

#임퓨터를 사용하여 filled_train으로 저장 이후 같은 임퓨터를 사용할때는 imputer.transform()으로 사용하면됨
filled_data = imputer.fit_transform(usr_data)

#사용하면 array값으로 나오기때문에 dataframe으로 바꿔주고 컬럼을가져옴
filled_data = pd.concat([app_id_data, pd.DataFrame(filled_data)], axis = 1)

# 다시 원래 열 이름 가져온다.
filled_data.columns = col_names

# 역 라벨 인코딩 실시.
for i in label_col:
    inv_map = {v: k for k, v in label_dict[i].items()}
    filled_data[i] = usr_data[i].map(inv_map)


# 정상적으로 NA가 채워졌는지 확인.

filled_data.isna().sum()

# 작업시작 6시 54분(PM) 22.09.27 -> 작업 종료 02시 55분(AM) 22.09.28 (총 8시간.)

filled_data.to_csv('full_user_spec+_fillna.csv', index = False)


# k 최적화도 가능한데. 이거 언제 끝나는지 감이 안와서 안함. 따이
# https://wikidocs.net/125444
# verbose 메서드 미아. -> 랜덤포레스트로 메우는 방법도 존재한다. 실험적인 옵션.
# https://velog.io/@juyeonma9/%EA%B2%B0%EC%B8%A1%EC%B9%98-%EC%B2%98%EB%A6%AC

'''
#%%

# user_log_plus 전처리 -> no log 처리방안 + 변수 제거
usr_log_plus.columns

# 사용하지 않을 변수는 drop
usr_log_plus = usr_log_plus.drop(['event', 'mp_os','mp_app_version','timestamp'], axis = 1)

# latest_os,latest_version에 등장하는 no_log는 unknown 처리.
usr_log_plus.latest_os = usr_log_plus.latest_os.replace('no log', 'unknown')
usr_log_plus.latest_version = usr_log_plus.latest_version.replace('no log', 'unknown')

# 나머지 no_log는 0처리
usr_log_plus = usr_log_plus.replace('no log', 0)

# latest_os는 대소문자가 이상하게 퍼저있다. - 소문자화.
usr_log_plus.latest_os = usr_log_plus.latest_os.str.lower()

# latest_version은 아예 좀 미친듯이 범주가 이상하다.
# 일단 건드리지 않는다. 독립적으로 이해한다.


# 남는 부분의 no_log는 치환하여 준다.
usr_log_plus = usr_log_plus.replace('no log', 0)


#%%
# usr_data와 log데이터를 합친다. 그 뒤 결측값 추가 처리 예정.
# 이때 user_id는 제거하고 합친다.
filled_data = usr_data # 앞서서 KNN 새로 완료한 파일을 받았다.

filled_data = filled_data.astype({'application_id' : int, 'user_id': float})
usr_log_plus = usr_log_plus.drop(['user_id'], axis = 1)
filled_data = pd.merge(filled_data, usr_log_plus, how = 'left', on = 'application_id')

del log_data

# 형 변환시켜줄것
filled_data = filled_data.astype({'update_times' : float, 'GetCreditInfo' : float, 'UseLoanManage' : float, 'Login' : float,
                                  'OpenApp' : float, 'UsePrepayCalc' : float, 'StartLoanApply' : float, 'ViewLoanApplyIntro' : float,
                                  'CompleteIDCertification' : float, 'EndLoanApply' : float, 'SignUp' : float, 'UseDSRCalc' : float})


#%%

# 이제 진짜 완전한 데이터 프레임이 완성되었다. >> ㄴㄴㄴㄴ 전처리 다합치고 쓸것.
# filled_data = pd.read_csv('full_user_spec+log_fillna.csv')

full_data = pd.merge(rst_data, filled_data, how = 'left', on = 'application_id')

print(full_data.isna().sum())

full_data.to_csv('full_data.csv', index = False)



# 전처리 마지막 단계로 loan_limit / rate만 예측하면 되는 상황


#%%
# 약 30분 소요.
full_data = pd.read_csv('full_data.csv')
na_filled_rst_data =  pd.read_csv('loan_result_fillna.csv')

# train기간의 full data 결측은 제거
# test 기간의 결측은 loan_result_fillna 방법에 의거해 채운 방법으로 보간.

# test기간에만 결측 보간.
test_len = ['2022-06-'+str(i).zfill(2) for i in range(1,31)]

for idx in tqdm(full_data[full_data.loan_limit.isna()].index):
    temp = full_data.loc[idx]
    if temp.loanapply_insert_time[:10] in test_len:
        appid = temp.application_id
        bkid = temp.bank_id
        pdid = temp.product_id
        aptime = temp.loanapply_insert_time
        
        line = na_filled_rst_data[(na_filled_rst_data.application_id == appid) & (na_filled_rst_data.bank_id == bkid) &
                                  (na_filled_rst_data.product_id == pdid) & (na_filled_rst_data.loanapply_insert_time == aptime)]
        full_data.loc[idx, 'loan_limit'] = line.loan_limit.values
        full_data.loc[idx, 'loan_rate'] = line.loan_rate.values
    
# 확인해보면 6월 이후는 전부 메워짐
# aa = full_data[full_data.loan_limit.isna()]


full_data_test = full_data[full_data.is_applied.isna()]
full_data_train = full_data[~(full_data.is_applied.isna())]

# test기간에 결측무
print(full_data_test.isna().sum())

# train 기간에 결측존재. = drop
full_data_train.isna().sum()
full_data_train = full_data_train.dropna()

full_data_train.to_csv('N:/[03] 단기 작업/빅콘 테스트/full_data_train_2.csv', index = False)
full_data_test.to_csv('N:/[03] 단기 작업/빅콘 테스트/full_data_test_2.csv', index = False)

# test 범주 안에서 제외된 샘플 존재 X
# complete 데이터 만든다음, test기간 길이 같은지 반드시 볼것. 빠진 요소 있으면 안됨.
print(rst_data.is_applied.isna().sum())
print(full_data_test.shape)

# 즉 이제서야 데이터 클리닝 작업이 완료되었음.



