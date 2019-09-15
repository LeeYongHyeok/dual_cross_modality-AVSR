import os
import sys
import subprocess
db_list = ['pretrain','train','val','test']
LRS2_path = '/home/data/bell_DB/[DB]_LIPREADING/LRS2/'
LRS2_wav_path = '/home/data/bell_DB/[DB]_LIPREADING/LRS2_wav/'
con_path = '/home/data/bell_DB/[DB]_LIPREADING/LRS_con/'
con_wav_path = '/home/data/bell_DB/[DB]_LIPREADING/LRS_con_wav/'

for db in db_list:
    print("concat : ", db)

    with open(LRS2_path + db + '.txt', 'r') as file:
        for line in file.readlines():
            item = line.strip().split('\t')
            if db == 'pretrain': 
                source_wav = LRS2_wav_path + db + '/' + item[0] + '.wav'
                dest_wav = con_wav_path + db + '/' + item[0] + '.wav'
                
                source_mp4 = LRS2_path + db + '/' + item[0] + '.mp4'
                dest_mp4 = con_path + db + '/' + item[0] + '.mp4'
                
                source_txt = LRS2_path + db + '/' + item[0] + '.txt'
                dest_txt_2_wav = con_wav_path + db + '/' + item[0] + '.txt'
                dest_txt_2_mp4 = con_path + db + '/' + item[0] + '.txt'
    
                dest_path_2_wav = dest_txt_2_wav[0:-9]
                dest_path_2_mp4 = dest_txt_2_mp4[0:-9]
            
            if db == 'train' or db == 'val':
                source_wav = LRS2_wav_path + 'main' + '/' + item[0] + '.wav'
                dest_wav = con_wav_path + 'trainval' + '/' + item[0] + '.wav'
                
                source_mp4 = LRS2_path + 'main' + '/' + item[0] + '.mp4'
                dest_mp4 = con_path + 'trainval' + '/' + item[0] + '.mp4'
                
                source_txt = LRS2_path + 'main' + '/' + item[0] + '.txt'
                dest_txt_2_wav = con_wav_path + 'trainval' + '/' + item[0] + '.txt'
                dest_txt_2_mp4 = con_path + 'trainval' + '/' + item[0] + '.txt'
    
                dest_path_2_wav = dest_txt_2_wav[0:-9]
                dest_path_2_mp4 = dest_txt_2_mp4[0:-9]

            if db == 'test':
                tmp = item[0]
                source_wav = LRS2_wav_path + 'main' + '/' + tmp[0:-3] + '.wav'
                dest_wav = con_wav_path + 'test' + '/' + tmp[0:-3] + '.wav'
                
                source_mp4 = LRS2_path + 'main' + '/' + tmp[0:-3] + '.mp4'
                dest_mp4 = con_path + 'test' + '/' + tmp[0:-3] + '.mp4'
                
                source_txt = LRS2_path + 'main' + '/' + tmp[0:-3] + '.txt'
                dest_txt_2_wav = con_wav_path + 'test' + '/' + tmp[0:-3] + '.txt'
                dest_txt_2_mp4 = con_path + 'test' + '/' + tmp[0:-3] + '.txt'
    
                dest_path_2_wav = dest_txt_2_wav[0:-9]
                dest_path_2_mp4 = dest_txt_2_mp4[0:-9]

            if not os.path.isdir(dest_path_2_mp4):
                subprocess.call('mkdir {}'.format(dest_path_2_mp4), shell=True)
            if not os.path.isdir(dest_path_2_wav):
                subprocess.call('mkdir {}'.format(dest_path_2_wav), shell=True)

            subprocess.call('cp {} {}'.format(source_wav, dest_wav), shell=True)
            subprocess.call('cp {} {}'.format(source_mp4, dest_mp4), shell=True)
            subprocess.call('cp {} {}'.format(source_txt, dest_txt_2_mp4), shell=True)
            subprocess.call('cp {} {}'.format(source_txt, dest_txt_2_wav), shell=True)
