# -*- coding: utf-8 -*- 

# 영상 -> 오디오  |  json 에서 발화시작, 종료시간별로 오디오 클립 분할 + bad data제거 (bad sampled)
#
#
# sample rate, encoding error 거르는 전처리예시 https://github.com/homink/speech.ko


'''
1. json frame별로 오디오 클립 분할
2. bad data제거
'''

# y index 넣어서 start frame ~
# scipy
# meta data : wave path, emotion label 저장

# matplotlib for displaying the output
import matplotlib.pyplot as plt


# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
from pydub import AudioSegment
import preprocess.json_to_mel
import os
import glob
import preprocess.tf_record

def transforma(audio_path):
    # audio_path list 받아서 for 문으로 넣을 수 있게

    y, sr = librosa.load(audio_path)
    y_save.append(y)
    sr_save.append(sr)

    return y_save, sr_save


def audiocut(audio_path, start, end, output_path):
    newAudio = AudioSegment.from_wav(audio_path)
    newAudio = newAudio[start:end]  # milisecond 단위
    newAudio.export(output_path, format="wav")

def save_numpy(save_path, data_path):
    # 저장할 path 만들기
    labels = ['0', '1', '2', '3', '4', '5', '6']  # NEEDS to add more label like
    # 0=anger, 1=disgust&contempt, 2=afraid, 3=happiness, 4=sadness, 5=surprise, 6=neutral

    if os.path.exists(os.path.join(save_path, labels)) == False:
        os.makedirs(os.path.join(save_path, labels))

    # data_path 를 순차적으로 part1/KETI_MULTIMODAL_0000000012_334_235345 등을 돌게 지정해줘야함
    raw_data_path = ''
    data_path_name_list = []
    data_path_list = []

    for j in range(3):
        part_path = os.path.join(raw_data_path, 'part', str(j))
        data_path_name_list = os.path.dirname(part_path)

        for i in range(len(data_path_name_list)):
            data_path_list.append(os.path.join(part_path, data_path_name_list[i], data_path_name_list[i]))
            preprocess.json_to_mel.read(save_path, data_path_list[i])



if __name__ == '__main__':
    data_path = '/Users/stella/dev/korean-audio-sentiment-analysis/data/raw'  # 자동화
    save_path = '/Users/stella/dev/korean-audio-sentiment-analysis/data/train'
    save_numpy(save_path, data_path)


    SR = 16000
    fs = SR // 1000

    preprocess.tf_record.create_tfrecords(save_path)
