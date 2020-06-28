import librosa
import os 
import wave
from scipy.io.wavfile import write
import numpy as np
import json
import codecs
from random import uniform


#utils for audio
'''
-load_wav
-path walk through
-remove silence
 묵음제거 https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/blob/master/audio/silence.py
    
-duration to frame
-frame to duration
    https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/blob/master/audio/get_duration.py
-audio augmentation
'''
sample_rate = 16000
def get_duration(audio):
    return librosa.core.get_duration(audio, sr=sample_rate)
def save_audio(audio, path, sample_rate=None):
    audio *= 32767 / max(0.01, np.max(np.abs(audio)))
    librosa.output.write_wav(path, audio.astype(np.int16),
                             sample_rate if sample_rate is None else sample_rate)

    print(" [*] Audio saved: {}".format(path))

def load_wav(wav_path):
 y, sr = librosa.load(wav_path)
 duration = librosa.get_duration(y=y, sr=sr)
 print("duration",duration)
 return y

def walk_dir(path):
 list = [] 
 for root, dirs, files in os.walk(path):
  for file in files:
   list.append(os.path.join(root,dirs,file))
 return list


def get_silence(sec):
    return np.zeros(sample_rate * sec)

def load_audio(path, pre_silence_length=0, post_silence_length=0):
    audio = librosa.core.load(path, sr=sample_rate)[0]
    if pre_silence_length > 0 or post_silence_length > 0:
        audio = np.concatenate([
            get_silence(pre_silence_length),
            audio,
            get_silence(post_silence_length),
        ])
    return audio
   
 
def load_and_trim(wav_path, TOP_DB):
 y, sr = librosa.load(wav_path)
 yt, index = librosa.effects.trim(y, top_db=TOP_DB)
 return yt
  
  
def remove_breath(audio):
    edges = librosa.effects.split(
            audio, top_db=40, frame_length=128, hop_length=32)

    for idx in range(len(edges)):
        start_idx, end_idx = edges[idx][0], edges[idx][1]
        if start_idx < len(audio):
            if abs_mean(audio[start_idx:end_idx]) < abs_mean(audio) - 0.05:
                audio[start_idx:end_idx] = 0

    return audio

def split_on_silence_with_librosa( #묵음을 기준으로 파일 분리
        audio_path, top_db=40, frame_length=1024, hop_length=256,
        skip_idx=0, out_ext="wav",
        min_segment_length=3, max_segment_length=8,
        pre_silence_length=0, post_silence_length=0):

    filename = os.path.basename(audio_path).split('.', 1)[0]
    in_ext = audio_path.rsplit(".")[1]

    audio = load_audio(audio_path)

    edges = librosa.effects.split(audio,
            top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    new_audio = np.zeros_like(audio)
    for idx, (start, end) in enumerate(edges[skip_idx:]):
        new_audio[start:end] = remove_breath(audio[start:end])
        
    save_audio(new_audio, add_postfix(audio_path, "no_breath"))
    audio = new_audio
    edges = librosa.effects.split(audio,
            top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    audio_paths = []
    for idx, (start, end) in enumerate(edges[skip_idx:]):
        segment = audio[start:end]
        duration = get_duration(segment)

        if duration <= min_segment_length or duration >= max_segment_length:
            continue

        output_path = "{}/{}.{:04d}.{}".format(
                os.path.dirname(audio_path), filename, idx, out_ext)

        padded_segment = np.concatenate([
                get_silence(pre_silence_length),
                segment,
                get_silence(post_silence_length),
        ])


        
        save_audio(padded_segment, output_path)
        audio_paths.append(output_path)

    return audio_paths

def metaread(filename, time):
    read = wave.open(filename, 'r')

    #get sample rate
    frameRate = read.getframerate()

    #get number of frames
    numFrames = read.getnframes()

   #get duration
    duration = numFrames/frameRate

    #get all frames as a string of bytes
    frames = read.readframes(numFrames)

   #get 1 frame as a string of bytes
    oneFrame = read.readframes(1)

    #framerate*time == numframesneeded
    numFramesNeeded=frameRate*time

    #numFramesNeeded*oneFrame=numBytes
    numBytes = numFramesNeeded*oneFrame   
 
    return frameRate,numFrames,duration,frames,oneFrame,numFramesNeeded,numBytes

‘’’
usage
convert_pitch(‘past.wav’,’new.wav’,1)
convert_pitch(‘past.wav’,’new.wav’,-1)
‘’’
def convert_pitch(wav_path, save_path, pitch_step):
    pitch_factor = 2**(pitch_step/12) * 22050
    speed_factor = 2**(-pitch_step/12)
    os.system('ffmpeg -i {} -af "asetrate=r={}, atempo={}, aresample=24000" {}'.format(wav_path, pitch_factor, speed_factor, save_path))

 def convert_audio_speed(wav_path, save_path, speed_factor):
     '''
     speed_factor = 0.8 면 0.8배 속도
     '''
     os.system('ffmpeg -i {} -af atempo={} {}'.format(wav_path, speed_factor, save_path))   

