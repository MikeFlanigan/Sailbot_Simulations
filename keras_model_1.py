from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.layers import Input, Dense, Dropout
from keras.models import Model

# fix random seed for reproducible testing
##seed = 17
##np.random.seed(seed)

# load data
X = np.load('X_1.npy')
Y = np.load('Y_1.npy')

# scale normalize X
for i in np.arange(X.shape[1]):
    print('Max val in feature',i,X[:,i].max())
    X[:,i] = X[:,i]/X[:,i].max()
    print('Max val in feature',i,X[:,i].max())

# shuffling data
indx = np.arange(X.shape[0]) 
np.random.shuffle(indx)
X = X[indx]
Y = Y[indx]

# Splitting data into train, validation, and test sets
def SplitData(splits,X,Y): # splits is a list of three %'s that sum to 1
    splits = [int(np.floor(X.shape[0]*splits[0])),
              int(np.floor(X.shape[0]*splits[1])),
              int(np.floor(X.shape[0]*splits[2]))]
    X_train = X[0:splits[0],:]
    Y_train = Y[0:splits[0],:]

    X_val = X[splits[0]:splits[0]+splits[1],:]
    Y_val = Y[splits[0]:splits[0]+splits[1],:]

    X_test = X[splits[0]+splits[1]:X.shape[0],:]
    Y_test = Y[splits[0]+splits[1]:X.shape[0],:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

splits = [.9,.05,.05] # splits is a list of three %'s that sum to 1
X_train, Y_train, X_val, Y_val, X_test, Y_test = SplitData(splits,X,Y)

print('Training set shape:',X_train.shape)
print('Testing set shape:',X_test.shape)
print('Validation set shape:',X_val.shape)


# custom evaluation function for increased clarity/insight into model performance
def custom_eval(dec_thresh,predictions,labels):
        # classification evaluation
        true_pos = len(np.where((predictions>.5) & labels == 1)[0])
        false_pos = len(np.where((predictions[0]<.5) & labels == 1)[0])
        true_neg = len(np.where((predictions[0]<.5) & labels == 0)[0])
        false_neg = len(np.where((predictions[0]<.5) & labels == 1)[0])

        if (true_pos+false_pos)>0:
                precision = true_pos/(true_pos+false_pos)
        else:
                precision = 0
        if (true_pos+false_neg) > 0:
                recall = true_pos/(true_pos+false_neg)
        else:
                recall = 0
        if (precision+recall)>0:
                F1score = 2*precision*recall/(precision+recall)
        else:
                F1score = 0
  
        class_results = [F1score,precision,recall,true_pos,false_pos,true_neg,false_neg]
        return class_results
    


# make a custom callback since regular tends to prints bloat notes
class cust_callback(keras.callbacks.Callback):
        def __init__(self):
                self.train_loss = []
                self.val_loss = []
        def on_epoch_end(self,epoch,logs={}):
                print('epoch:',epoch,' loss:',logs.get('loss'),' validation loss:',logs.get('val_loss'))
                self.val_loss.append(logs.get('val_loss'))
                self.train_loss.append(logs.get('loss'))

                plt.clf()
                plt.ion()
                plt.plot(self.train_loss, 'b',label='train loss')
                plt.plot(self.val_loss, 'r',label='val loss')
                plt.legend()
                plt.draw()
                plt.pause(0.01)

                return
        def on_batch_end(self,batch,logs={}):

                return
history = cust_callback()

# define model
##model = Sequential()
##model.add(Dense(96, input_dim=24, init='uniform', activation='relu'))
##model.add(Dropout(0.2))
##model.add(Dense(120, init='uniform', activation='relu'))
##model.add(Dropout(0.2))
##model.add(Dense(75, init='uniform', activation='relu'))
##model.add(Dropout(0.2))
##model.add(Dense(30, init='uniform', activation='relu'))
##model.add(Dropout(0.2))
##model.add(Dense(6, init='uniform', activation='sigmoid'))

# new model define to support 2x 1/2 softmaxing
XData = Input(shape = (X_train.shape[1],))
denX = Dense(96,init='uniform', activation='relu')(XData)
denX = Dropout(0.2)(denX)
denX = Dense(120, init='uniform', activation='relu')(denX)
denX = Dropout(0.2)(denX)
denX = Dense(75, init='uniform', activation='relu')(denX)
denX = Dropout(0.2)(denX)
denX = Dense(30, init='uniform', activation='relu')(denX)
denX = Dropout(0.2)(denX)
##soft1 = Dense(3, init='uniform', activation='softmax')(denX)
##soft2 = Dense(3, init='uniform', activation='softmax')(denX)
##outputDa = keras.layers.merge([soft1,soft2], mode='concat', concat_axis=1)
outputDa = Dense(6,init = 'uniform',activation = 'linear')(denX)
model = Model([XData],[outputDa])
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.save('sailbot_t1.h5')
sys.exit()
try:
    # Fit the model
    model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=16,nb_epoch=500, verbose = 0, callbacks=[history])
except KeyboardInterrupt:
    pass

plt.clf()
plt.ion()
plt.plot(history.train_loss, 'b',label='train loss')
plt.plot(history.val_loss, 'r',label='val loss')
plt.legend()
plt.show()

def Evaluate():
    # evaluate the model
    dec_thresh = 0.5
    print('TRAINING SET RESULTS:')
    predictions = model.predict(X_train) # make predicitions
    class_results = custom_eval(dec_thresh,predictions,Y_train)
    F1score,precision,recall,true_pos,false_pos,true_neg,false_neg = class_results
    print('True Pos:',true_pos,'False Pos:',false_pos,'True Neg:',true_neg,'False Neg:',false_neg)
    print('F1score:',F1score)

    print('VALIDATION SET RESULTS:')
    predictions = model.predict(X_val) # make predicitions
    class_results = custom_eval(dec_thresh,predictions,Y_val)
    F1score,precision,recall,true_pos,false_pos,true_neg,false_neg = class_results
    print('True Pos:',true_pos,'False Pos:',false_pos,'True Neg:',true_neg,'False Neg:',false_neg)
    print('F1score:',F1score)

    print('TEST SET RESULTS:')
    predictions = model.predict(X_test) # make predicitions
    class_results = custom_eval(dec_thresh,predictions,Y_test)
    F1score,precision,recall,true_pos,false_pos,true_neg,false_neg = class_results
    print('True Pos:',true_pos,'False Pos:',false_pos,'True Neg:',true_neg,'False Neg:',false_neg)
    print('F1score:',F1score)
Evaluate()

model.save('sailbot_t1.h5')
