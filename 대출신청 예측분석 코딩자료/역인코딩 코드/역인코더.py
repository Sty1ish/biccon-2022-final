# default setting
import os
import pickle

# data tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler


os.chdir(r'C:\Users\Nyoths\Documents\카카오톡 받은 파일\역인코딩 코드')

#%%
#표준화 값 역 인코딩용 코드.
with open('categorical_names.pickle', 'rb') as f:
    categorical_names = pickle.load(f)
with open('std_enc.pickle', 'rb') as f:
    std_enc = pickle.load(f)
with open('input_example.pickle', 'rb') as f:
    input_example = pickle.load(f)

feature_names = input_example.index

cat_feature_names = ['bank_id', 'product_id', 'gender', 'income_type', 'employment_type',
                     'houseown_type', 'purpose', 'latest_os', 'latest_version', 'personal_rehabilitation']
categorical_features = [idx for idx, val in enumerate(feature_names) if val in cat_feature_names]
num_features_names = [val for idx, val in enumerate(feature_names) if val not in cat_feature_names]
numeric_features = [idx for idx, val in enumerate(feature_names) if val not in cat_feature_names]


#%%

# 값 설정.

loan_limit = 0.47
loan_rate = 0.68
credit_score = 0.48
yearly_income = -0.13
desired_amount = -0.18
existing_loan_cnt = -0.91
existing_loan_amt = 0
update_times = 0
GetCreditInfo = 0
UseLoanManage = 0
Login = -0.10
OpenApp = -0.16
UsePrepayCalc = 0
StartLoanApply = -0.27
ViewLoanApplyIntro = -0.10
CompleteIDCertification = 0
EndLoanApply = -0.30
SignUp = 0
UseDSRCalc = -0.07
employ_date = 0
age = -0.64

arr = [loan_limit, loan_rate, credit_score, yearly_income, desired_amount, existing_loan_cnt, existing_loan_amt, update_times, GetCreditInfo, UseLoanManage, Login, OpenApp, UsePrepayCalc, StartLoanApply, ViewLoanApplyIntro, CompleteIDCertification, EndLoanApply, SignUp, UseDSRCalc, employ_date, age]


inverse_df = pd.DataFrame([num_features_names, arr, std_enc.inverse_transform(np.array(arr).reshape(1,-1)).reshape(-1,)])

inverse_df


# inverse_df가 우리 데이터셋 상수 데이터 inverse_transform 한 결과.

std_enc.scale_
std_enc.mean_

# 당연하지만 (값 * scale[표준편차]) + mean[뮤] = 역인코딩이고
# 순서는 arr에 등장해있다.

# loan_limit 역계산.
(0.33 * std_enc.scale_[0]) + std_enc.mean_[0]


# loan_rate 역계산.
(0.78 * std_enc.scale_[1]) + std_enc.mean_[1]

# credit_score 역계산
(-1.03 * std_enc.scale_[2]) + std_enc.mean_[2]
