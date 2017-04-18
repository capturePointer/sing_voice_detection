from baseZhang import init_data_dir
import os
from tqdm import tqdm
from scipy.io.wavfile import read

dir_path = init_data_dir()
print dir_path  # "../data/sing_voice_detection/"
mir1k_wav_data = "../data/sing_voice_detection/Wavfile"


def get_right_voice_channle(data_dir="../data/sing_voice_detection/test"):
    for root, dir_name, files in os.walk(data_dir):
        for audio_file in tqdm(files):
            audio_path = os.path.join(root, audio_file)
            [fs, data] = read(audio_path)
            print data, fs
            print type(data)
    return 0
get_right_voice_channle()