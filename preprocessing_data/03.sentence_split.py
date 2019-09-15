import sys
import os
from pydub import AudioSegment
from tqdm import tqdm
import shutil

wav_root = "/home/data/bell_DB/[DB]_LIPREADING/LRS3_wav"
subfolder_list = ["pretrain"]
write_root = "/home/data/bell_DB/[DB]_LIPREADING/LRS3_wav_word"

#if os.path.isdir(write_root):
#    shutil.rmtree(write_root)

#os.makedirs(write_root)


for subfolder in subfolder_list:
    subfolder_write_root = str(os.path.join(write_root,subfolder))
    subfolder_root = str(os.path.join(wav_root,subfolder))
    #os.makedirs(subfolder_write_root)
    folder_list = os.listdir(subfolder_root)

    for utt in folder_list:
        utt_root = str(os.path.join(subfolder_root,utt))
        file_list = os.listdir(utt_root)
        utt_write_root = str(os.path.join(subfolder_write_root,utt))
        #os.makedirs(utt_write_root)
        
        for wavfile in file_list:
            ext = wavfile.split(".")[-1]
            if ext != "wav":
                continue

            wavfile_write_root = str(os.path.join(utt_write_root,wavfile))
            wavfile_root = str(os.path.join(wav_root,subfolder,utt,wavfile))
            textfile_root = wavfile_root.replace(".wav",".txt")
            print(wavfile_root)
            print(wavfile_write_root)
            print(textfile_root)
            sys.exit()
