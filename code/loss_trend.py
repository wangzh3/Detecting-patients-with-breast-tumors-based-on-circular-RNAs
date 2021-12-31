from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

df=pd.read_csv("history.csv").values
plt.figure("loss")
x=[]
for i in range(len(df[:,1])):
    x.append(i+1)
l1=plt.plot(x,df[:,1],c='b',label='train')
l2=plt.plot(x,df[:,2],c='r',label='validation')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim((1,20))
plt.title("loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig("loss.jpg")

df=pd.read_csv("/Users/jfv472/Documents/mywork/cRNA/cmpt830/selected_data.csv").values
label=df[:,1]

#len=6480
test_x_normal=df[100:114,2:] #14
test_y_normal=label[100:114]

test_x_tumor=df[1114:,2:] #135
test_y_tumor=label[1114:]

def getdata(x_normal, x_tumor, y_normal, y_tumor):
    x = np.vstack((x_normal, x_tumor)).astype(np.int)
    y = np.append(y_normal, y_tumor).astype(np.int)
    return x,y

def reshape_onehot(x,y):
    x1=x.reshape(x.shape[0],1,x.shape[1],1)
    y1 = y.reshape(len(y), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(y1)
    return x1,onehot_encoded

x_test,y_test=getdata(test_x_normal, test_x_tumor, test_y_normal, test_y_tumor)
x_test1, y_test1 = reshape_onehot(x_test, y_test)

model=load_model('mymodel.h5')
predict=model.predict(x_test1)
predict1=[np.argmax(item) for item in predict]

mat=confusion_matrix(y_test,predict1)
print(mat)
scores = model.evaluate(x_test1,y_test1, verbose=1)
print("loss: ",scores[0])
print("acc: ",scores[1])
