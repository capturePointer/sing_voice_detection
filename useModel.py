import os
import time
import joblib
import numpy
import math
import numpy as np
from baseZhang import calcMFCC, load_model, load_model_weights
from keras.utils import np_utils
from scipy.io.wavfile import read


SEED = 1007
np.random.seed(SEED)
LOAD_MODEL_FLAG = True
EPOCH = 10000
TIME_SIZE = 7
ENCODER = joblib.load('encoder.joblib')

def filter_each_frame(feature, audio_owner, TIME_SIZE):
    temp_feature = []
    dataset_X = []
    dataset_Y = []
    for feature_frame in feature:
        drop_flag = False
        for number in feature_frame:
            if math.isinf(number) or math.isnan(number):
                drop_flag = True
                break
        if not drop_flag:
            temp_feature.append(feature_frame)
    for item in range(0, len(temp_feature) - TIME_SIZE, TIME_SIZE):
        dataset_X.append(temp_feature[item:item + TIME_SIZE])
        dataset_Y.append(audio_owner)
    return dataset_X, dataset_Y

def load_lstm_model(onDataset='dataset.h5', featureX='X'):
    onDataset_name = onDataset.split('/')[-1].split('.')[0]
    model_save_path = 'models/lstm_' + featureX + '_' + onDataset_name + '_model.json'
    model_weights_save_path = model_save_path.replace('.json', '.h5')

    model = load_model(model_save_path)
    model = load_model_weights(model_weights_save_path, model)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])  # accuracy mae fmeasure precision recall
    return model


def wav_preprocess(audio_path):
    mfcc_dataset_X = []
    mfcc_dataset_Y = []
    win_size = 1024
    TIME_SIZE = 7
    label_tag = ''
    if '_left_music.wav' in audio_path or '_mix_vocal_music.wav' in audio_path or '_right_vocal.wav' in audio_path:
        if '_left_music.wav' in audio_path:
            label_tag = 'onlyMusic'
        elif '_mix_vocal_music.wav' in audio_path:
            label_tag = 'withVocal'
        elif '_right_vocal.wav' in audio_path:
            label_tag = 'onlyVocal'
        else:
            pass
        # mfcc
        fs, signal = read(audio_path)
        mfcc_feature = calcMFCC(signal, fs, win_length=win_size / 1.0 / fs, win_step=win_size / 1.0 / fs)
        mfcc_X, mfcc_Y = filter_each_frame(mfcc_feature, label_tag, TIME_SIZE)
        mfcc_dataset_X.extend(mfcc_X)
        mfcc_dataset_Y.extend(mfcc_Y)

    return mfcc_dataset_X, mfcc_dataset_Y


def predict_singer_max(wav_path, model):
    X_test, Y_test = wav_preprocess(wav_path)

    Y_predict = model.predict(X_test)
    Y_predict = np_utils.probas_to_classes(Y_predict)
    Y_predict = ENCODER.inverse_transform(Y_predict)
    Y_predict = list(Y_predict)
    Y_predict = max(Y_predict, key=Y_predict.count)

    return Y_predict


def predict_singer_prob(wav_path, model):
    X_test, Y_test = wav_preprocess(wav_path)
    Y_predict = model.predict(X_test)
    Y_predict = numpy.array([map(sum, zip(*Y_predict))])
    Y_predict = np_utils.probas_to_classes(Y_predict)
    Y_predict = ENCODER.inverse_transform(Y_predict)

    return Y_predict[0]


def predict_frame_tag(wav_path, model):
    X_test, Y_test = wav_preprocess(wav_path)
    Y_predict = model.predict(X_test)
    Y_predict = np_utils.probas_to_classes(Y_predict)
    Y_predict = ENCODER.inverse_transform(Y_predict)

    return Y_predict


def song_level_acc(data_dir):
    total = 0
    right = 0
    useDataset = '../data/sing_voice_detection/dataset.h5'
    model = load_lstm_model(useDataset)
    for root, dir_name, files in os.walk(data_dir):
        for audio_file in files:
            audio_path = os.path.join(root, audio_file)
            label_tag = ''
            if '.wav' in audio_path:
                print "===========================================".replace('=', '>')

                if '_left_music.wav' in audio_path or '_mix_vocal_music.wav' in audio_path or '_right_vocal.wav' in audio_path:
                    if '_left_music.wav' in audio_path:
                        label_tag = 'onlyMusic'
                    elif '_mix_vocal_music.wav' in audio_path:
                        label_tag = 'withVocal'
                    elif '_right_vocal.wav' in audio_path:
                        label_tag = 'onlyVocal'
                    else:
                        pass
                    print"audio_tag:", label_tag
                    total += 1
                    predict_tag = predict_singer_prob(audio_path, model)
                    print"predict tag:", predict_tag
                    if label_tag == predict_tag:
                        right += 1
                    else:
                        print audio_path, '*-*'
                    print "==========================================="

    acc = right / 1.0 / total

    return acc


def frame_level_acc(data_dir):
    total = 0
    right = 0
    useDataset = '../data/sing_voice_detection/dataset.h5'
    model = load_lstm_model(useDataset)
    for root, dir_name, files in os.walk(data_dir):
        for audio_file in files:
            audio_path = os.path.join(root, audio_file)
            label_tag = ''
            if '.wav' in audio_path:
                if '_left_music.wav' in audio_path or '_mix_vocal_music.wav' in audio_path or '_right_vocal.wav' in audio_path:
                    print "===========================================".replace('=', '>')
                    if '_left_music.wav' in audio_path:
                        label_tag = 'onlyMusic'
                    elif '_mix_vocal_music.wav' in audio_path:
                        label_tag = 'withVocal'
                    elif '_right_vocal.wav' in audio_path:
                        label_tag = 'onlyVocal'
                    else:
                        pass
                    print"audio_tag:", label_tag

                    predict_tag = predict_frame_tag(audio_path, model)
                    for tag in predict_tag:
                        total += 1
                        if label_tag == tag:
                            right += 1
                        else:
                            print audio_path, '*-*'

                    print"predict tag:", predict_tag

                    print "==========================================="

    acc = right / 1.0 / total

    return acc


def main():
    wav_mir1k = "../data/sing_voice_detection/Wavfile/"
    # print frame_level_acc(wav_artist_20_clipsfull_dir_test)#0.863744439331
    print  song_level_acc(wav_mir1k)  # 0.97
    return 0


if __name__ == '__main__':
    print "start..."
    before = time.time()
    main()
    after = time.time()
    print 'takes time : %.0f(s)' % (after - before)
