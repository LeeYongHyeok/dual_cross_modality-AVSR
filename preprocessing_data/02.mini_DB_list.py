import os
import sys
import shutil
from tqdm import tqdm

read_folder = "/home/data/LIPREADING/scp/LRS_con/sentence/"
write_folder = "/home/data/LIPREADING/scp/LRS_con/pre_train_test/"
subfolder_list = ["pretrain", "trainval"]

if os.path.isdir(write_folder):
    shutil.rmtree(write_folder)
if not os.path.isdir(write_folder):
    os.makedirs(write_folder, exist_ok=True)

# pretrain, test, trainval
num_speaker = [9999, 9999, 0]
max_length = 160

for subfolder in subfolder_list:
    if num_speaker[subfolder_list.index(subfolder)] == 0:
        continue

    # initialization
    prev_spk = ""
    num_spk = 0
    cut_path = write_folder + subfolder + "_" + str(num_speaker[subfolder_list.index(subfolder)])

    w = open(cut_path, "a+")
    full_DB_path = read_folder + subfolder
    r = open(full_DB_path)

    # read text file
    line = r.readline()
    while line:
        utt_id = str(line).split(' ', maxsplit = 2)[0]
        curr_spk = utt_id.split('/')[1]
        
        # if find different spk, add num_spk
        if not prev_spk == curr_spk:
            num_spk += 1
            prev_spk = curr_spk
        
        if num_spk == num_speaker[subfolder_list.index(subfolder)] + 1:
            break
        
        w.write(line)
        line = r.readline()
    r.close()
    w.close()
     
    command = "awk 'length($0)<" + str(max_length + 20) + "' " + cut_path + " | awk '{ print length, $0  }' | sort -n -s | cut -d \" \" -f2-> " + write_folder + subfolder + "_sort_character_labels"
    print(command)
    os.system(command)
    command = "cat " + write_folder + subfolder + "_sort_character_labels"+ " | awk '{ print $1 }' > " + write_folder + subfolder + "_sort.scp"
    os.system(command)

