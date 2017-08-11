import numpy as np
np.set_printoptions(threshold=np.inf)
import keras
from keras.layers import Dense, Input, concatenate, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
import random

#max label 179
#max frame 3236
#frames (163392, 1)
#units (163392, 1600, 6)
#buildings (163392, 800, 8)
#labels (163392, 1)
# * .2 = 32678
    
def genData(frames, units, buildings, labels, its, bs):
    all_ixs = [i for i in range(len(labels))]
    random.shuffle(all_ixs)
    for i in range(0,int(163392) * its,bs):
        ixs = all_ixs[i:i+bs]
        x = []
        y = np.zeros((bs,180))
        arr = np.zeros((bs,1600,6))
        for k, ix in enumerate(ixs):
            arr[k] = units[ix]
        x.append(arr)

        arr = np.zeros((bs,800,8))
        for k, ix in enumerate(ixs):
            arr[k] = buildings[ix]
        x.append(arr)
        arr = np.zeros((bs,1))
        for k, ix in enumerate(ixs):
            arr[k] = frames[ix]
        x.append(arr)
        for k, ix in enumerate(ixs):
            y[k][int(labels[ix][0])] = 1
        yield (x, y)
    return

def run(path, model, its, bs):
    frames = np.load(path+'frame.npy', 'r')
    units = np.load(path+'unit_features.npy', 'r')
    buildings = np.load(path+'building_features.npy', 'r')
    labels = np.load(path+'labels.npy', 'r')

    model.fit_generator(genData(frames, units, buildings, labels, its, bs),
                        int(163392/bs), epochs=its, verbose=1)

    for d in genData(frames, units, buildings, labels, its, bs):
        out = model.predict(d[0], batch_size=bs, verbose=0)
        for pred in out:
            print(np.where(pred == 1)+' : '+np.where(d[1] == 1))
        break
    
    return
    
def makeModel():
    inputs = []

    model = Input(shape=(1600,6))
    inputs.append(model)
    u_model = TimeDistributed(Dense(6, activation='relu'))(model)
    u_model = Flatten()(model)

    model = Input(shape=(800,8))
    inputs.append(model)
    b_model = TimeDistributed(Dense(8, activation='relu'))(model)
    b_model = Flatten()(model)

    frame_input = Input(shape=(1,))
    inputs.append(frame_input)
    
    f_model = Dense(1, activation='relu')(frame_input)
    u_model = Dense(2000, activation='relu')(u_model)
    b_model = Dense(1500, activation='relu')(b_model)

    model = concatenate([u_model, b_model, f_model])
    model = Dense(3500, activation='relu')(model)
    out = Dense(180, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def testModel():
    inp = Input(shape=(6,))
    out = Dense(6, activation='relu')(inp)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

if __name__=='__main__':
    its = 1
    batch_size = 256
    path = 'C:\\Users\\kbruhwiler\\Downloads\\clean_data\\'
    model = makeModel()
    #model = testModel()
    run(path, model, its, batch_size)
