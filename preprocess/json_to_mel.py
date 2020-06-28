# -*- coding: utf-8 -*- 

import json
import io
import numpy as np
import os
import librosa
import pandas as pd
import pickle
import cv2
import pprint
import pandas as pd
# matplotlib for displaying the output
import matplotlib.pyplot as plt
import argparse
import librosa
import librosa.display
import glob



class json_to_mel :
    def __init__(self):
        pass

    def read(self, file_path):
        file_path_real = glob.glob(file_path)[0]
        print(file_path_real)
        with io.open(file_path_real, encoding='utf-8') as f:
            #io.open(f, encoding='utf-8-sig')
            #raw_data = yaml.safe_load(f)
            raw_data = json.load(f, encoding ='utf-8')
            #raw_data = json.load(f, object_hook=_decode_dict)
        return raw_data

    def parser(self, json_data,mp4_path): #parsing
        #print(mp4_path)
        cap = cv2.VideoCapture(mp4_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(length) 
        duration = length/fps

        dialogue_df = pd.DataFrame()
        idx = 0
        for i in json_data['dialogue_infos']:
            dialogue = {'speaker_id':i['speaker_id'], 'start_time':i['start_time'], 'end_time':i['end_time'], 'duration':duration}
            data = pd.DataFrame(dialogue, index=[idx])
            dialogue_df = dialogue_df.append(data)
            idx += 1

        df = pd.DataFrame()
        idx = 0
        for i in json_data['shot_infos']:
            if len(i['visual_infos'][0]['persons']) <= 0:
                continue
    
            speaker = i['visual_infos'][0]['persons'][0]['person_id']
            frames = []
            for j in i['visual_infos']:
                frames.append(j['frame_id'])
                emotion = j['persons'][0]['person_info']['emotion']
            emotion.update({'frame_id':j['frame_id'], 'frame_length':length, 'person_id':speaker, 'first_frame':min(frames), 'last_frame':max(frames)})
            data = pd.DataFrame(emotion, index=[idx])
            df = df.append(data)
            idx += 1

        #pickle 로 저장
        #log S 와 비교해서 list, array 중에 format 결정할것.
        npy_save = np.array(df)
        # with open(parse_save_path, 'wb', pickle.HIGHEST_PROTOCOL) as wow:
        #     pickle.dump(df, wow)
    
        npy_save_dialogue = np.array(dialogue_df)

        return npy_save, npy_save_dialogue

    def str_to_sec(self, str_list):
        temp = []
        for str_val in str_list:
            if isinstance(str_val, str):
                h, m, s = str_val.split(':')
                temp.append(int(h) * 3600 + int(m) * 60 + int(float(s)))
            else:
                print(str_val)
            
        return temp

    #{frame_id:, emotion}
    def split_by_emo(self, file_path, save_path, npy_save, npy_save_dialogue):
        #file_path : json path
        #data path : mp4 등 저장된 위치
        #save path : 결과물 저장될 위치
        #npy_save parser 로 부터 넘겨받는 데이터

        # file_list = os.listdir(data_path)
        # file_list_json = [file for file in file_list if file.endswith(".json")]
        #

        real_file_name = file_path[:-19]  #327_interpolation.json -> 327 #NEEDS change based on datasets
        audio_name = real_file_name + '.mp4'
        y, sr = librosa.load(audio_name)

        label = []
        all_arr = []

        # emo = npy_save['emotion']
        # all_real_fr_num = npy_save['frame_length'] # 정현님이 주시는 전체 프레임수
        # frame_info_s = npy_save['first_frame']
        # frame_info_e = npy_save['last_frame']
        #all_real_time_num = npy_save_dialogue['duration']
        emo = np.argmax(npy_save[:,:8],axis=1)
        all_real_fr_num = npy_save[:,9]
        all_real_time_num = npy_save_dialogue[:,3]

        frame_info_s = npy_save[:,11]
        frame_info_e = npy_save[:,12]

        fract = len(y)/all_real_fr_num # 프레임수(이미지)에 대한 y 의 비율
        fract2 = len(y)/all_real_time_num # 시간에 대한 y 의 비율

        # time_info_s = npy_save_dialogue['start_time']
        # time_info_e = npy_save_dialogue['end_time']

        time_info_s = self.str_to_sec(self.npy_save_dialogue[:,1])
        time_info_e = self.str_to_sec(self.npy_save_dialogue[:,2])

        for k in range(0, npy_save_dialogue.shape[0]):
            now_e = emo[k]   #emotion 라 dic key emotion
            now_ss = frame_info_s[k]*fract
            now_es = frame_info_e[k]*fract
            now_ss_2 = time_info_s[k]*fract2
            now_es_2 = time_info_e[k]*fract2

           #TODO if 문에서 에러가 나네용
           if now_ss >= now_ss_2 :
                if now_es <= now_es_2 :

                    label.append(now_e)
                    now_l = now_e

                    # 0=anger, 1=disgust&contempt, 2=afraid, 3=happiness, 4=sadness, 5=surprise, 6=neutral

                    # if (now_e == 'anger'):
                    #     label.append(0)
                    #     now_l = 0
                    # if (now_e == 'disgust'):
                    #     label.append(1)
                    #     now_l = 1
                    # if (now_e == 'contempt'):
                    #     label.append(1)
                    #     now_l = 1
                    # if (now_e == 'afraid'):
                    #     label.append(2)
                    #     now_l = 2
                    # if (now_e == 'happiness'):
                    #     label.append(3)
                    #     now_l = 3
                    # if (now_e == 'sadness'):
                    #     label.append(4)
                    #     now_l = 4
                    # if (now_e == 'surprise'):
                    #     label.append(5)
                    #     now_l = 5
                    # if (now_e == 'neutral'):
                    #     label.append(6)
                    #     now_l = 6

                    all_arr.append(y[now_ss:now_es])
                    S = librosa.feature.melspectrogram(np.asarray(y[now_ss:now_es]), sr=sr, n_mels=128)

                    # Convert to log scale (dB). We'll use the peak power (max) as reference.
                    # Check the Log S format
                    log_S = librosa.power_to_db(S, ref=np.max)

                    # 여기서 라벨에 따라서 save path 를 변경
                    #TODO 파일명 변경
                    this_save_path = os.path.join(save_path,now_l,real_file_name+'_'+time_info_s[k]+ '_' + time_info_e[k] + '.png') #save to datasets/0/KETI_MULTIMODAL_0000000012_334_235345

                    plt.switch_backend('agg')
                    # Make a new figure
                    plt.figure(figsize=(12, 4))

                    # Display the spectrogram on a mel scale
                    # sample rate and hop length parameters are used to render the time axis
                    librosa.display.specshow(log_S, sr=sr, x_axis='off', y_axis='off')

                    # Put a descriptive title on the plot
                    # plt.title('mel power spectrogram')

                    # draw a color bar
                    # plt.colorbar(format='%+02.0f dB')

                    # Make the figure layout compact
                    plt.tight_layout()
                    plt.savefig(this_save_path)
        #TODO label만 리턴하고 이미지에 파일명 정보가 있게끔 하거나 label별로 파일을 만들어서 이미지를 저장해도 됩니당.
        return log_S, label

        
if __name__ == '__main__':

    '''
    usage 
    python json_to_mel.py --data_path RAWDATA_PATH --save_path PROCESSED_PATH
    '''
    
    #data_path : json, mp4 가 있는 경로
    #save_path : label과 mel을 각각 pickle로 저장할 경로
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",default='../sample', help="data_path")
    parser.add_argument("--save_path",default='../preprocessed', help="data_path")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    preprocesser = json_to_mel()

    abs_data_path = os.path.abspath(args.data_path)
    abs_save_path = os.path.abspath(args.save_path)



    # json_path = os.path.join(abs_data_path,'*.json' )
    # mp4_path = os.path.join(abs_data_path,'*.mp4' )
    file_list = os.listdir(abs_data_path)
    file_list_json = [file for file in file_list if file.endswith(".json")]
    for single_file in file_list_json :
        mp4_path = single_file[:-19]+'.mp4'
        clean_data_frame, clean_data_time = preprocesser.parser(preprocesser.read(single_file), mp4_path)
        clean_data_mel , clean_data_label = preprocesser.split_by_emo(single_file, abs_save_path, clean_data_frame, clean_data_time)



    # save_clean_path = os.path.join(abs_save_path,'clean.pkl' ) ## 바꾸기 2개로
    # save_mel_path = os.path.join(abs_save_path,'mel.pkl' )


      #clean_data_label.to_pickle(save_label_path)
    #clean_data_mel.to_pickle(save_mel_path)




