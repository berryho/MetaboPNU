# 2. Use the best model

from keras.models import load_model
from sklearn import preprocessing
import numpy as np
import pandas as pd


# data set
ud = pd.read_csv('../dataset/ginseng-example.csv')
Udataset = ud.values
unknown_set_ = Udataset[:,2:568]
Un_id = Udataset[:,0]
unknown_set = preprocessing.normalize(unknown_set_)



# load model
model = load_model('../code/stored_model/930-0.2492.hdf5')
model.summary()

# use model
unknown_pred = ud['Class'].unique()[model.predict_classes(unknown_set)]
unknown_percent = model.predict(unknown_set)
index = []
id_ = []
pred = []
per = []
for i in range(len(unknown_set)):
    prediction = unknown_pred[i]
    index.append(i+1)
    id_.append(Un_id[i])
    pred.append(unknown_pred[i])
    per.append(np.max(unknown_percent[i]))
predict_result = pd.DataFrame({'< ID >': id_, ' < predict >':pred, ' < percent >':per}, index=index)
print(predict_result.head(len(unknown_set)))
