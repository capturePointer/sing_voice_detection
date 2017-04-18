import os
import time

import joblib
import numpy as np
from baseZhang import save_model, save_model_weights, load_model, load_model_weights, if_no_create_it
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from preprocessData import get_dataset_X_Y_encoder

SEED = 1007
np.random.seed(SEED)
LOAD_MODEL_FLAG = False
EPOCH = 10000
TIME_SIZE = 7


# INPUT_DIM = 13  # 13 12 25 37
def load_train_test_data(onDataset, featureX='X', targetY='Y'):
    X, one_hot_Y, encoder = get_dataset_X_Y_encoder(onDataset, featureX, targetY)
    joblib.dump(encoder, 'encoder.joblib')
    split_to_int_0000 = len(X) / 10000
    X = X[:split_to_int_0000 * 10000]
    one_hot_Y = one_hot_Y[:split_to_int_0000 * 10000]
    X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_Y, test_size=0.2, random_state=SEED)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=SEED)
    return X_train, X_test, Y_train, Y_test, X_val, Y_val


def build_lstm_model(onDataset='dataset.h5', featureX='X', targetY='Y'):
    onDataset_name = onDataset.split('/')[-1].split('.')[0]
    X_train, X_test, Y_train, Y_test, X_val, Y_val = load_train_test_data(onDataset, featureX, targetY)
    print 'len Xtrain=================>>>>>>>>>>>>>>>>>>', len(X_train)
    INPUT_DIM = len(X_train[0][0])
    FEATRE_DIM = INPUT_DIM
    model_save_path = 'models/lstm_' + featureX + '_' + onDataset_name + '_model.json'
    if_no_create_it(model_save_path)
    model_weights_save_path = model_save_path.replace('.json', '.h5')
    if_no_create_it(model_weights_save_path)
    print 'weight:', model_weights_save_path
    if LOAD_MODEL_FLAG and os.path.isfile(model_save_path) and os.path.isfile(model_weights_save_path):
        print model_save_path
        model = load_model(model_save_path)
        model = load_model_weights(model_weights_save_path, model)
    else:
        # create model
        model = Sequential()
        model.add(
            LSTM(500, input_shape=( TIME_SIZE, FEATRE_DIM)))
        model.add(Dropout(.5))
        model.add(Dense(3, init='normal', activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])  # accuracy mae fmeasure precision recall
    # train model ...
    print "train..."
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, min_delta=0.02)
    # model.fit(X_train, Y_train, validation_split=0.2, verbose=2, callbacks=[earlyStopping], nb_epoch=EPOCH,
    #           batch_size=BATCH_SIZE)

    model.fit(X_train, Y_train, verbose=2, shuffle=True, callbacks=[earlyStopping], nb_epoch=EPOCH,
               validation_data=(X_val, Y_val))
    save_model(model, model_save_path)
    save_model_weights(model, model_weights_save_path)

    loss_and_metrics = model.evaluate(X_test, Y_test,  verbose=0)
    print '==============='
    print 'loss_metrics: ', loss_and_metrics

    return loss_and_metrics[1]


def main():
    acc = build_lstm_model(onDataset='../data/sing_voice_detection/dataset.h5')  # step 1. test on dnn model
    print acc
    return 0


if __name__ == '__main__':
    print "start..."
    before = time.time()
    main()
    after = time.time()
    print 'takes time : %.0f(s)' % (after - before)
