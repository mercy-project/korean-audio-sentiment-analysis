# korean-audio-sentiment-analysis_v2
This project is a part of the [mercy-project](https://github.com/mercy-project).

## Project Intro/Objective
이 프로젝트의 목적은 딥러닝 음성 기술을 이용해서 한국어 오디오에서 감정을 분류하는 것입니다.  
The purpose of this project is to classify emotions in Korean audio using deep learning voice technology.
 * sentiment classification (speech 2 class)

## Technologies
* tf.keras 2.x

## Data
Multimodel data http://aihub.or.kr/aidata/135
Korean free speech audio data http://aihub.or.kr/aidata/105

## Directory Layout

    ├── preprocess                
          ├── audio_utils.py         # load wav
          ├── json_to_mel.py         # extract annotated sentiment label from json
          ├── preprocessing.py       # extract mel spectrogram from audio, save tf record
          ├── tf_record.py           # tf record util
          └── raw_data_parser.ipynb  # extract mel spectrogram and label from sample data
    ├── sample                       # sample data
          ├── KETI_MULTIMODAL_0000000012.mp4
          └── KETI_MULTIMODAL_0000000012_interpolation.json
    ├── base_knowledge               # documentation files for preprocessing, sentiment analysis
          ├── paper 
              └──Multi-Modal Emotion recognition on IEMOCAP.md 
          └── preprocessing 
             ├── dft_fft_stft_mfcc             
                  ├── DFT_FFT_STFT.md          
                  └── MFCCs.md              
             └── stft_mfcc_melspectrogram
                  ├──Librosa_stft_mfcc_melspectrogram.ipynb
                  └──STFT_MFCC_Melspectrogram.md
         
    ├── dataloader.py                # preprocess data as tf record
    ├── evaluate.py                  # evaluate model
    ├── keras_train.py               # baseline training codes for pickle data
    ├── model.py                     # baseline model with fine-tunable inception v3
    ├── train.py                     # training codes from pickle data
    ├── train_model.ipynb.           # baseline trainining codes from sample data using inception v3
    └── README.md


## About base knowledge materials

base knowledge for audio sentiment analysis
  1) preprocessing : in audio preprocessing, there are many methods
  which can analyze in frequency domain. we search some ways and summarized the
  characteristic features.
  * keywords : dft, fft, stft, mfcc 
  
  2) paper : we research about the sentiment classification and find the paper which is 
  related to our project. 
  * Multi-Modal Emotion recognition on IEMOCAP with neural networks


## Getting Started
1. Install requirements
```shell
python
pandas
matplotlib
librosa
opencv-python
tensorflow
argparse
numpy
```
2. Download Raw data from AIhub(http://aihub.or.kr/aidata/135)
3. Preprocessing with [codes](preprocess/raw_data_parser.ipynb)

|    sentiment metadata        | dialogue metadata      | 
|------------|-------------| 
|<img src="https://github.com/Ella77/korean-audio-sentiment-analysis/blob/master/image/sample_sentiment.png" width="500"> |  <img src="https://github.com/Ella77/korean-audio-sentiment-analysis/blob/master/image/sample_dialogue.png" width="500"> |


visualized spectrogram from each annotated emotions 

<img src='https://github.com/Ella77/korean-audio-sentiment-analysis/blob/master/image/sample_mel.png' width=500 />

4. Training the model [codes](train_model.ipynb)



## Contributing Members
|Name     |   Link          | 
|---------|-----------------|
|[박정현](https://github.com/parkjh688)| @parkjh688       |
|[양서연](https://github.com/howtowhy) |     @howtowhy   |
|[강수진](https://github.com/Ella77) |     @Ella77   |
