import tensorflow as tf
import numpy as np
import glob
import os

 # get item 가져올때마다, dataloader

class tf_record :

    def _bytes_features(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _int64_features(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def write_tfrecords(self, tfrecord_path, sounds, labels):
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for sound, label in zip(sounds, labels):
      

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'signal_raw': self._bytes_features(sound),
                        'label': self._int64_features(label)
                    }))
                writer.write(example.SerializeToString())


    def create_tfrecords(self, save_path):
        # tfrecord_pathes = pathes for train, val tfrecords

        sounds, labels = [], []

        label_list = os.path.basename(save_path)

        for now_label in label_list:       #walk through dir

            wav_path_list = os.path.join(save_path, now_label)
            wav_path = os.path.dirname(wav_path_list)
            for wav in wav_path :
                feature = np.load(wav)
                #random seed and split into train and val data
                sounds.append(feature)
                labels.append(now_label)

            #print(len(sounds), len(labels))
            tfrecord_path = os.path.join(save_path, now_label, 'tfrecord')
            print('Writing tfrecords...')
            self.write_tfrecords(tfrecord_path, sounds, labels)

# pytorch vesion class BaseDataset(Dataset):
#     def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
#         self.wav_paths = wav_paths
#         self.labels
#         self.bos_id, self.eos_id = bos_id, eos_id

#     def __len__(self):
#         return len(self.wav_paths)

#     def count(self):
#         return len(self.wav_paths)

#     def getitem(self, idx):
#         #trim process 가져오기 (datautils에서)
#         feat = get_mfcc(self.wav_paths[idx])
#         label = get_label(self.script_paths[idx])
#         return feat, label

