import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
np.random.seed(1)


data = np.genfromtxt('http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', delimiter=',', dtype='<U20')
labels = data[:,0]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:]

# 첫 변수가 라벨(예측변수)
# class_names은 예측값 라벨
# data는 예측변수를 제외한 라벨. 22개

# 전체 변수가 categorical
categorical_features = range(22) 

# 전체 열 이름(data의 열 이름.)
feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(',')


#%%

categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
fibrous=f,grooves=g,scaly=y,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises=t,no=f
almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
attached=a,descending=d,free=f,notched=n
close=c,crowded=w,distant=d
broad=b,narrow=n
black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
enlarging=e,tapering=t
bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
fibrous=f,scaly=y,silky=k,smooth=s
fibrous=f,scaly=y,silky=k,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
partial=p,universal=u
brown=n,orange=o,white=w,yellow=y
none=n,one=o,two=t
cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')
for j, names in enumerate(categorical_names):
    values = names.split(',')
    values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
    data[:,j] = np.array(list(map(lambda x: values[x], data[:,j])))
    
#즉 요약된 데이터셋이, 원래의 형태로 돌아옴(이름으로)
# 원레 데이터셋에선 이 작업은 불필요하네..
#%%

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_
    
#%%
data[:,0]


#%%
# 이상태에서 데이터셋은 전부 라벨 인코딩 진행됨.
data = data.astype(float)

#%%
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)


#%%
encoder = sklearn.preprocessing.OneHotEncoder()
encoder.fit(data)
encoded_train = encoder.transform(train)

#%%
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(encoded_train, labels_train)

#%%

predict_fn = lambda x: rf.predict_proba(encoder.transform(x))
sklearn.metrics.accuracy_score(labels_test, rf.predict(encoder.transform(test)))

#%%
explainer = lime.lime_tabular.LimeTabularExplainer(train ,class_names=['edible', 'poisonous'], feature_names = feature_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)
i = 137
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.as_pyplot_figure()