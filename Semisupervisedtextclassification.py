import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

dataset=pd.read_csv("IT comments.csv")
commentVector=dataset["comment"].values
vec=CountVectorizer(stop_words='english')
tf=TfidfTransformer()
a=vec.fit_transform(commentVector)
tfComments=tf.fit_transform(a)

y=dataset["IT"].values
ytrain=y
ytrain[10000:]=-1

labelPropModel = LabelPropagation(gamma=10)
labelPropModel.fit(tfComments.toarray(),ytrain)
print(labelPropModel.score(tfComments.toarray(),y))
