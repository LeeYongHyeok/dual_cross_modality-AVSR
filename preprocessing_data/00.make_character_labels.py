import os
import shutil
from tqdm import tqdm
import sys
dir_root = "/home/data/LIPREADING/raw_data/LRS_concat/LRS_con_wav_crop_0.5_from_mp4"
subfolder_list = ["pretrain_subvid_0.5"]
write_folder = "/home/data/LIPREADING/scp/LRS_con/"

if not os.path.isdir(write_folder):
    os.makedirs(write_folder, exist_ok=True)

charcter_label_path = write_folder + "/character_labels"
if os.path.isfile(charcter_label_path):
    os.remove(charcter_label_path)


f = open(charcter_label_path, "a+")

for subfolder in subfolder_list:
    if os.path.isfile(write_folder + "/" + subfolder):
        os.remove(write_folder + "/" + subfolder)
    print(subfolder)
    sub = open(write_folder + "/" + subfolder, "a+") 
    utt_list = os.listdir(os.path.join(dir_root,subfolder))
    for utt in tqdm(utt_list):
        file_list = os.listdir(os.path.join(dir_root, subfolder, utt))

        for txt_file in file_list:
            txt_path = str(os.path.join(dir_root, subfolder, utt, txt_file))
            ext = txt_path.split('.')[-1]

            if ext != "txt":
                continue

            txtfile = open(txt_path, "r")
            txt_line = txtfile.readline()
            txtfile.close()
            
            # LRS3 label style "Text:  IT HAD A STRONG STURDY ..."
            pre = txt_line[0:7]
            label = txt_line.split(pre)[1]
            #label = txt_line[7:-1]
#            if label[-1] == " ":
#                new_label = label[0:-1]
#                #label = new_label + "<EOS>"
#                label = new_label 
#            else:
#                #label = label + "<EOS>"
#                label = label
            utt_id = str(os.path.join(subfolder,utt,txt_file.split('.')[0]))
            charcter_label = utt_id + " " + label + "\n"
            f.write(charcter_label)
            sub.write(charcter_label)
    sub.close()
f.close()
