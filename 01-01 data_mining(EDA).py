#%%
# default setting
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'C:\Users\styli\Desktop\빅콘 테스트')

rst_data = pd.read_csv('loan_result.csv')
log_data = pd.read_csv('log_data.csv')
usr_data = pd.read_csv('user_spec.csv')


#%%
######################
##### 결측치 체크 #####
######################
rst_data.isna().sum() # 총 13,527,363건
# loan_limit / loan_rate 7,495건 결측
# 애는 같은 application id의 회귀 결과로 해결 보는게 맞을것 같음.
len(rst_data.application_id.unique()) # 968,866건 적어도 건당 10~12장은 나온다는 뜻이니까 회귀 해도 될듯.
len(rst_data[rst_data.loan_limit.isna() == True].application_id.unique()) #결측이 나온 애들의 id는 쏠린게 아니라 한두개씩 빠짐.

# loan_limit가 결측인 애들이, loan_rate가 결측인것으로 확인 = 동시결측
rst_data[rst_data.drop('is_applied',axis=1).loan_limit.isnull()].equals(rst_data[rst_data.drop('is_applied',axis=1).loan_rate.isnull()])
# 그에 반해 TARGET이 NA인지 아닌지는 무관함.

# TRAIN과 TARGET에 동시 존재하는 application_id
len(set(rst_data[rst_data.is_applied.isna() == True].application_id.unique()).intersection(set(rst_data[rst_data.is_applied.isna() == False].application_id.unique())))

# TARGET값인 is_applied 3257239건 (test 데이터임. 분할 안된 데이터셋)
rst_data[rst_data.is_applied.isna() == True] # test 데이터셋 3,257,239건 
rst_data[rst_data.is_applied.isna() == False] # train 데이터셋 10,270,123건




log_data.isna().sum() # 총 17,843,993건
# 딱히 결측에 특징은 없어보이는 정보들. 고유 유저 기준 최빈값으로 바꾸면 될?듯?
# mp_os 980건
# mp_app_version 660,597건



usr_data.isna().sum() # 총 1,394,216건
len(usr_data.user_id.unique()) # 고유 유저의 수는 405,213건 / 결측을 이걸로 채우는게 맞나? 신청 당시와 지금은 조건이 바뀌지 않음?
# 고유 유저 기준 결측값 채울수 있는 애들
# birth_year / gender 12,961건
usr_data[usr_data.birth_year.isnull()].equals(usr_data[usr_data.gender.isnull()]) # 생년월일과 성별은 동시결측이 맞다.
# 생년월일은 나이로 바껴야하고, 범주화 작업하던가 해야할듯 + gender는 군집화/knn보간 가능.

# credit_score 105115건 (이거 좀 골치아프네)


# 결측이 0일수도 있는 변수들.(취직을 안했다던가 등등...)
# yearly_income 90건
# income_type 85건
# company_enter_month 171760건
# employment_type / houseown_type / purpose 85건
# personal_rehabilitation_yn 587461건
# personal_rehabilitation_complete_yn 1203354건
# existing_loan_cnt 198556건
# exsisting_loan_amt 313774건


# 왜 결측이 난지 1도 모르겠는 애들.
# / desired_amount 85건 / 영향이 없다면 분석 제외 소망... 안될 가능석 큼.





# 합치기 전 처리? or 합친 후 처리? 일단 합쳐놓고 생각.
# 로그 데이터는 전처리 이후에 유저 단위로 붙여야 할 것으로 생각함.

#%%
# usr_data의 결측부분 해결하기.


# EDA 선행.
# 팀 판단 이후 결정


#%%

# log data 전처리
counter = 0
for idx, df in log_data.groupby('user_id'):
    try:
        pass
    except:
        # 에러는 따로 출력해서 확인하고
        print('에러 발생 id와 df')
        print(idx); print(df)
    finally:
        pass
    # 출력값 반환받고 끗.




#%%
########################
##### 데이터 합치기 #####
########################

# 일단 전처리 이전 데이터셋 완성. > log_data는 따로 붙여야할것.
data = pd.merge(rst_data, usr_data, how='left', on='application_id')


#%%
# rst_data의 결측은 회귀 모델로 해결 >> usr_data를 완성하고 난다음 진행해야 할것같음.

counter = 0
for idx, df in data.groupby('application_id'):
    try:
        if df.isna().sum().sum() == 0:
            break # 결측이 없으면 바로 합쳐주고
        else: # 결측이 있다면
            # rst_train
            # rst_test
            # 여기서 회귀 훈련하고 결측값 채운 모델을 만들고.
            pass
    except:
        # 에러는 따로 출력해서 확인하고
        print('에러 발생 id와 df')
        print(idx); print(df)
    finally:
        if counter == 0:
            counter = 1
            treat_rst_data = df
        else:
            treat_rst_data = pd.concat(['treat_rst_data', 'df'], axis = 0)
    # 출력값 반환받고 끗.
