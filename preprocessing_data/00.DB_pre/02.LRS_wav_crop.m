% clear all; close all;
function LRS3_wav_crop_final(timeTH)
data_path = '/home/data/LIPREADING/raw_data/LRS_concat/LRS_con_wav';
% data_path = 'data';
makeMode='pretrain';
% makeMode='test';
% subFolder=makeMode;
warning('off');
timeTH=0.5;

save_root = ['/home/data/LIPREADING/raw_data/LRS_concat/LRS_con_wav_crop_' num2str(timeTH)];

fid_label=fopen(sprintf('character_label_%0.1f',timeTH),'w');

folderList=dir(fullfile(data_path,makeMode));
folderList=folderList(3:end);
% figure;
temp_idx=1;
sub_wav_list=zeros(10000,2);
sub_label_list=cell(10000,1);
for i=1:numel(folderList)
% for i=1
    fileList=dir(fullfile(folderList(i).folder,folderList(i).name, '*.txt'));
    wavList=dir(fullfile(folderList(i).folder,folderList(i).name, '*.wav'));
       
    savingFolder = fullfile(save_root, makeMode, folderList(i).name);
    mkdir(savingFolder); 

    for j=1:numel(fileList)
%     for j=1
        fid=fopen(fullfile(fileList(j).folder,fileList(j).name));
        [wav, fs] = audioread(fullfile(wavList(j).folder,wavList(j).name));
        
        audio_name=split(fileList(j).name,'.');
        sub_idx=0;
        disp([num2str(i) '/' num2str(numel(folderList)) ' ' fullfile(wavList(j).folder,wavList(j).name)]);
        % skip dummy
        for k=1:5
            tline = fgetl(fid);
        end
        next_ok=1;
        while ischar(tline)
%             disp(tline);
            sub_idx= sub_idx+1;
            word=split(tline,' ');
            start_time = str2double(word{2});
            end_time = str2double(word{3});
            while (end_time - start_time) < timeTH
                tline = fgetl(fid);
                if ischar(tline)
                    word_temp=split(tline,' ');
                    word{1} = [word{1} ' ' word_temp{1}];
                    end_time = str2double(word_temp{3});
                else
                    next_ok=0;
                    break;
                end
            end
            if next_ok || sub_idx ==1
                sub_wav_list(sub_idx,1)=max(1,floor(start_time*fs));
                sub_wav_list(sub_idx,2)=min(size(wav,1),floor(end_time*fs));
                sub_label_list{sub_idx,1}=sprintf('%s/%s/%s_%05d %s\n', makeMode, folderList(i).name, audio_name{1}, sub_idx, word{1});
                old_word=word{1};
                old_start_time=start_time;
                old_end_time=end_time;
            else
	            sub_idx=sub_idx-1;
				new_word=[old_word ' ' word{1}];
	            start_time = old_start_time;
		        sub_wav_list(sub_idx,1)=max(1,floor(start_time*fs));
			    sub_wav_list(sub_idx,2)=min(size(wav,1),floor(end_time*fs));
				sub_label_list{sub_idx,1}=sprintf('%s/%s/%s_%05d %s\n', makeMode, folderList(i).name, audio_name{1}, sub_idx, new_word);
            end
            
            tline = fgetl(fid);
        end
        fclose(fid);
        
        for k=1:sub_idx
            if (sub_wav_list(k,2)-sub_wav_list(k,1)>= fs*timeTH) || k==1
				audiowrite(fullfile(savingFolder,sprintf('%s_%05d.wav',audio_name{1}, k)),wav(sub_wav_list(k,1):sub_wav_list(k,2),:),fs);
	            fprintf(fid_label,sub_label_list{k});
			end 
        end
        
    end
end
fclose(fid_label);
% load(fullfile(dataroot,fileName));




