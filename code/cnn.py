from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import confusion_matrix
import random
df=pd.read_csv("/Users/jfv472/Documents/mywork/cRNA/cmpt830/selected_data.csv").values
label=df[:,1]

#len=6480
train_x_normal=df[0:100,2:] #100
train_y_normal=label[0:100]

test_x_normal=df[100:114,2:] #14
test_y_normal=label[100:114]

train_x_tumor=df[114:1114,2:] #1000
train_y_tumor=label[114:1114]

test_x_tumor=df[1114:,2:] #135
test_y_tumor=label[1114:]

def check_intersection(A,B):
    a = set((tuple(i) for i in A))
    b = set((tuple(i) for i in B))
    return len(a.intersection(b))

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

def argument(x):
    x1=np.copy(x)
    for i in range(x1.shape[0]):
        row=x1[i]
        idx=np.argwhere(row>0).flatten()
        j=random.choice(idx)
        x1[i][j]+=1
    return x1

start1=np.copy(train_x_normal)
start2=np.copy(train_y_normal)
#data arguentation
for i in range(9):

    train_x_normal = np.vstack((train_x_normal, argument(start1))).astype(np.int)
    train_y_normal=np.append(train_y_normal, start2).astype(np.int)


x_train,y_train=getdata(train_x_normal, train_x_tumor, train_y_normal, train_y_tumor)
x_test,y_test=getdata(test_x_normal, test_x_tumor, test_y_normal, test_y_tumor)
print(x_train.shape)
print(x_test.shape)
print(check_intersection(x_train,x_test))

num_classes = 2
batch_size = 128
epochs =50
seed = 7

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cvscores10 = []
mat10=[]

for i in range(1):
    cvscores = []
    train_loss = []
    val_loss = []
    print("\n %d experiments: " %(i+1))
    model = Sequential()
    ## *********** First layer Conv
    model.add(Conv2D(32, kernel_size=(1, 81), strides=(1, 1),
                     input_shape=(1, 6480, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(1, 2))
    ## ********* Classification layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # default lr=0.001
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    callbacks = [EarlyStopping(monitor='categorical_accuracy', patience=2, verbose=1)]
    for idx_train, idx_val in kfold.split(x_train, y_train):
        input_x, input_y = reshape_onehot(x_train[idx_train], y_train[idx_train])
        val_x, val_y = reshape_onehot(x_train[idx_val], y_train[idx_val])

        history = model.fit(input_x, input_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=callbacks,
                            verbose=1,
                            validation_data=(val_x, val_y))
        print("history")
        print(history.history['loss'])
        print(history.history['val_loss'])
        train_loss=train_loss+history.history['loss']
        val_loss=val_loss+history.history['val_loss']

        scores = model.evaluate(val_x, val_y, verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        predict = model.predict(val_x)
        predict1 = [np.argmax(item) for item in predict]
        print(confusion_matrix(y_train[idx_val], predict1))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    cvscores10.append(np.mean(cvscores))

    x_test1, y_test1 = reshape_onehot(x_test, y_test)
    predict = model.predict(x_test1)
    predict1 = [np.argmax(item) for item in predict]
    mat10.append(confusion_matrix(y_test, predict1))
    #save loss
    train_loss1=np.array(train_loss).reshape(-1,1)
    val_loss1 = np.array(val_loss).reshape(-1, 1)
    df=np.hstack((train_loss1,val_loss1))
    df1=pd.DataFrame(df,columns=['train_loss','val_loss'])
    csvfile = "/Users/jfv472/Documents/mywork/cRNA/cmpt830/history.csv"
    df1.to_csv(csvfile)
    #save model
    model.save("/Users/jfv472/Documents/mywork/cRNA/cmpt830/mymodel.h5")

for i in range(len(cvscores10)):
    print("%d exp's accuracy is %f" %(i+1,cvscores10[i]))
    print("confusion matrix:")
    print(mat10[i])
    print(" ")




