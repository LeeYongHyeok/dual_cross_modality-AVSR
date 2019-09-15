import os
import subprocess
import shutil

def search(d_name,li):
    for (paths, dirs, files) in os.walk(d_name):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.mp4':
                li.append(os.path.join(os.path.join(os.path.abspath(d_name),paths),filename)) 

READ_ROOT = '/home/data/LIPREADING/raw_data/LRS_concat/LRS_con_mp4/pretrain_subvid_1.2'

mp4_list = []
search(READ_ROOT,mp4_list)

for mp4_file in mp4_list:
    txt_file = mp4_file.replace(".mp4",".txt")
    wav_file = mp4_file.replace(".mp4",".wav")
    wav_file = wav_file.replace("LRS_con_mp4","LRS_con_wav_crop_1.2_from_mp4")
    wav_dir = wav_file.split(wav_file.split("/")[-1])[0]

    if os.path.isfile(wav_file):
        continue

    if not os.path.isdir(wav_dir):
        os.makedirs(wav_dir)

    command = 'ffmpeg -i {} {}'.format(mp4_file,wav_file)

    subprocess.call(command,shell=True)
    shutil.copy(txt_file,txt_file.replace("LRS_con_mp4","LRS_con_wav_crop_1.2_from_mp4"))
