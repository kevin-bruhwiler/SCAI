import numpy as np
np.set_printoptions(threshold=np.inf)
import keras
from keras.layers import Dense, Input, concatenate
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
        for r in range(1600):
            arr = np.zeros((bs,6))
            for k, ix in enumerate(ixs):
                arr[k] = units[ix][r].reshape((1,6))
            x.append(arr)
        for r in range(800):
            arr = np.zeros((bs,8))
            for k, ix in enumerate(ixs):
                arr[k] = buildings[ix][r].reshape((1,8))
            x.append(arr)
        arr = np.zeros((bs,1))
        for k, ix in enumerate(ixs):
            arr[k] = frames[ix].reshape((1,1))
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
    return
    
def makeModel():
    unit_inputs = []
    building_inputs = []
    inputs = []

    for i in range(1600):
        model = Input(shape=(6,))
        inputs.append(model)
        model = Dense(6, activation='relu')(model)
        unit_inputs.append(model)

    for i in range(800):
        model = Input(shape=(8,))
        inputs.append(model)
        model = Dense(8, activation='relu')(model)
        building_inputs.append(model)

    unit_model = concatenate(unit_inputs)
    building_model = concatenate(building_inputs)
    frame_input = Input(shape=(1,))
    inputs.append(frame_input)

    frame_model = Dense(1, activation='relu')(frame_input)
    unit_model = Dense(2000, activation='relu')(unit_model)
    building_model = Dense(1500, activation='relu')(building_model)

    model = concatenate([unit_model, building_model, frame_model])
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
    batch_size = 128
    path = 'C:\\Users\\kbruhwiler\\Downloads\\clean_data\\'
    model = makeModel()
    #model = testModel()
    run(path, model, its, batch_size)
