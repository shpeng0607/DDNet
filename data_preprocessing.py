import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pickle

class Config():
    def __init__(self):        
        self.frame_l = 32 # the length of frames
        self.joint_n = 22 # the number of joints
        self.joint_d = 3 # the dimension of classes        
        self.data_dir = 'DHG2016/'
        self.save_dir = 'DHG2016/'

C = Config()
data_list = np.loadtxt(C.data_dir +'informations_troncage_sequences.txt').astype('int16')

Train = {}
Train['pose'] = []
Train['coarse_label'] = []
Train['fine_label'] = []

Test = {}
Test['pose'] = []
Test['coarse_label'] = []
Test['fine_label'] = []

for i in tqdm(range(len(data_list))):
    idx_gesture = data_list[i][0]
    idx_finger = data_list[i][1]
    idx_subject = data_list[i][2]
    idx_essai = data_list[i][3]
    coarse_label = data_list[i][0]

    if data_list[i][1] == 1:
        fine_label = data_list[i][0]
    else:
        fine_label = data_list[i][0] + 14

    if idx_subject < 17:
        skeleton_path = C.data_dir + '/gesture_' + str(idx_gesture) + '/finger_' \
                    + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai)+'/'
            
        p = np.loadtxt(skeleton_path+'skeleton_world.txt').astype('float32')
        for j in range(p.shape[1]):
            p[:,j] = medfilt(p[:,j])
            
        Train['pose'].append(p)
        Train['coarse_label'].append(coarse_label)
        Train['fine_label'].append(fine_label)
    else:
        skeleton_path = C.data_dir + '/gesture_' + str(idx_gesture) + '/finger_' \
                + str(idx_finger) + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai)+'/'
        
        p = np.loadtxt(skeleton_path+'skeleton_world.txt').astype('float32')
        for j in range(p.shape[1]):
            p[:,j] = medfilt(p[:,j])
            
        Test['pose'].append(p)
        Test['coarse_label'].append(coarse_label)
        Test['fine_label'].append(fine_label)

print(len(Train['pose']))
print(len(Test['pose']))
pickle.dump(Train, open(C.save_dir+"train.pkl", "wb"))
pickle.dump(Test, open(C.save_dir+"test.pkl", "wb"))