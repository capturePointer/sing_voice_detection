import math
import os
import h5py
import numpy
from baseZhang import init_data_dir, calcMFCC
from keras.utils import np_utils
from scipy.io.wavfile import read, write
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

dir_path = init_data_dir()
print dir_path  # "../data/sing_voice_detection/"
mir1k_wav_data = "../data/sing_voice_detection/Wavfile"


def get_right_voice_channle(data_dir="../data/sing_voice_detection/test"):
    # setp 1. split music and vocal and get mix wav files
    for root, dir_name, files in os.walk(data_dir):
        for audio_file in tqdm(files):
            audio_path = os.path.join(root, audio_file)
            [fs, data] = read(audio_path)
            print data, fs
            print type(data)
            left = data[:, 0]
            right = data[:, 1]
            mix = left + right
            print left
            print right
            write(audio_path.replace('.wav', '_left_music.wav'), fs, left)
            write(audio_path.replace('.wav', '_right_vocal.wav'), fs, right)
            write(audio_path.replace('.wav', '_mix_vocal_music.wav'), fs, mix)

    return 0


# get_right_voice_channle(mir1k_wav_data)# setp 1. split music and vocal and get mix wav files

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


def get_train_test_data_set(data_dir="../data/sing_voice_detection/test"):
    # step 2. extract each frame mfcc feature and add to dataset X and Y
    mfcc_dataset_X = []
    mfcc_dataset_Y = []
    win_size = 1024
    TIME_SIZE = 7
    for root, dir_name, files in os.walk(data_dir):
        for audio_file in tqdm(files):
            label_tag = ''
            audio_path = os.path.join(root, audio_file)
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


def saveDataset(X, Y, nameX="X", nameY="Y", savePath="../data/SID/dataset.h5"):
    # step 3. save extracted dataset to local disk
    X = numpy.array(X)
    if os.path.isfile(savePath):
        file = h5py.File(savePath, 'a')
    else:
        file = h5py.File(savePath, 'w')
    file.create_dataset(nameX, data=X)
    file.create_dataset(nameY, data=Y)
    file.close()
    return 0


# mfcc_dataset_X, mfcc_dataset_Y = get_train_test_data_set(mir1k_wav_data)
# # step 2. extract each frame mfcc feature and add to dataset X and Y
# saveDataset(mfcc_dataset_X, mfcc_dataset_Y, "X", "Y", "../data/sing_voice_detection/dataset.h5")
# # step 3. save extracted dataset to local disk

def get_dataset_X_Y_encoder(dataset='dataset.h5', dataX='X', dataY='Y', encoder='None'):
    h5file = h5py.File(dataset, 'r')
    X = h5file[dataX][:]
    Y = h5file[dataY][:]
    h5file.close()
    if encoder == 'None':
        encoder = LabelEncoder()
        encoder.fit(Y)

    encoder_Y = encoder.transform(Y)
    # convert integers to variables (i.e. one hot encoded)
    one_hot_Y = np_utils.to_categorical(encoder_Y)
    return X, one_hot_Y, encoder
