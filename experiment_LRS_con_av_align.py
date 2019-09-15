import sys
import os
import avsr

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # ERROR


def main(argv):

    num_epochs = int(argv[1])
    learning_rate = float(argv[2])

    experiment = avsr.AVSR(
        unit='character',
        unit_file='/home/nas/user/yong/yong_Sigmedia-AVSR/datasets/lrs3/configs/character_list',
        video_processing='features',
        batch_normalisation=False,
        video_train_record='/home/nas/user/yong/LIPREADING/TFRecord/LRS_con/video/VGG_M_pre_05_av_clean.tfrecord',
        video_test_record='/home/nas/user/yong/LIPREADING/TFRecord/LRS_con/video/VGG_M_test_05_av_clean.tfrecord',
        audio_processing='features',
        audio_train_record='/home/nas/user/yong/LIPREADING/TFRecord/LRS_con/audio/lmda_90_pre_05_av_clean.tfrecord',
        audio_test_record='/home/nas/user/yong/LIPREADING/TFRecord/LRS_con/audio/lmda_90_test_05_av_clean.tfrecord',
        labels_train_record ='/home/nas/user/yong/LIPREADING/TFRecord/LRS_con/characters_pre_05_av.tfrecord',
        labels_test_record ='/home/nas/user/yong/LIPREADING/TFRecord/LRS_con/characters_test_05_av.tfrecord',
        encoder_type='unidirectional',
        architecture='dual_av_align_ga',
        clip_gradients=True,
        max_gradient_norm=1.0,
        recurrent_l2_regularisation=0.0001,
        cell_type='lstm',
        highway_encoder=False,
        sampling_probability_outputs=0.00,
        embedding_size=256,
        dropout_probability=(0.9, 0.9, 0.9),
        decoding_algorithm='beam_search',
        encoder_units_per_layer=((256, 256, 256), (256, 256, 256)),
        decoder_units_per_layer=(512,),
        attention_type=(('scaled_luong', )*1, ('scaled_luong', )*1),
        beam_width=10,
        batch_size=(128, 128),
        optimiser='Adam',
        learning_rate=learning_rate,
        num_gpus=1,
        write_attention_alignment=True,
    )

#    uer = experiment.evaluate(
#        checkpoint_path='/home/nas/user/yong/LIPREADING/logging/checkpoints/lrs_con_av_av_align_pre_05/',
#     )
#    print(uer)
#    return

    experiment.train(
        num_epochs=num_epochs,
        logfile='/home/nas/user/yong/LIPREADING/logging/logs/test',
        try_restore_latest_checkpoint=True
    )


if __name__ == '__main__':
    main(sys.argv)
