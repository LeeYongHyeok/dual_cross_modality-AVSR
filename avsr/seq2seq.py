import tensorflow as tf
from .encoder import Seq2SeqEncoder, AttentiveEncoder, AttentiveEncoder_test
from .decoder_bimodal import Seq2SeqBimodalDecoder
from .decoder_unimodal import Seq2SeqUnimodalDecoder
from .DualAttentiveEncoder import DualAttentiveEncoder

class Seq2SeqModel(object):
    def __init__(self,
                 data_sequences,
                 mode,
                 hparams):

        # member variables
        self._video_data = data_sequences[0]
        self._audio_data = data_sequences[1]

        self._mode = mode
        self._hparams = hparams

        self._make_encoders()
        self._make_decoder()
        self._init_saver()

    def _make_encoders(self):
        if self._hparams.architecture in ('va_align', 'va_align_ga'):
            if self._audio_data is not None:
                with tf.variable_scope('audio'):
                    self._audio_encoder = Seq2SeqEncoder(
                            data=self._audio_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                            )
            else:
                self._audio_encoder = None

            if self._video_data is not None:
                with tf.variable_scope('video'):
                    self._video_encoder = AttentiveEncoder(
                            data=self._video_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[0],
                            attended_memory=self._audio_encoder.get_data().outputs,
                            attended_memory_length=self._audio_data.inputs_length,
                            )
            else:
                self._video_encoder = None
            self._av_encoder = None
        else:
            if self._video_data is not None:
                with tf.variable_scope('av'):
                    if self._hparams.architecture in ('dual_av_align', 'dual_av_align_tr', 'dual_av_align_tr_ga', 'dual_av_align_ga'):
                        self._av_encoder = DualAttentiveEncoder(
                            audio_data=self._audio_data,
                            video_data=self._video_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            video_num_units_per_layer=self._hparams.encoder_units_per_layer[0],
                            audio_num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                        )
    
                    else:
                        self._av_encoder = None
    
                with tf.variable_scope('video'):
                    if self._hparams.architecture in ('unimodal', 'bimodal', 'av_align', 'av_align_add_vid_dec', 'av_align_test'):
                        self._video_encoder = Seq2SeqEncoder(
                            data=self._video_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[0],
                        )
    
            else:
                self._video_encoder = None
                self._av_encoder = None

            if self._audio_data is not None:
                with tf.variable_scope('audio'):
                    if self._hparams.architecture in ('unimodal', 'bimodal',):
                        self._audio_encoder = Seq2SeqEncoder(
                            data=self._audio_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                        )
                    elif self._hparams.architecture == 'av_align':
                        self._audio_encoder = AttentiveEncoder(
                            data=self._audio_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                            attended_memory=self._video_encoder.get_data().outputs,
                            attended_memory_length=self._video_data.inputs_length,
                        )
                    elif self._hparams.architecture == 'av_align_add_vid_dec':
                        self._audio_encoder = AttentiveEncoder(
                            data=self._audio_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                            attended_memory=self._video_encoder.get_data().outputs,
                            attended_memory_length=self._video_data.inputs_length,
                        )
                    elif self._hparams.architecture == 'av_align_test':
                        self._audio_encoder = AttentiveEncoder_test(
                            data=self._audio_data,
                            mode=self._mode,
                            hparams=self._hparams,
                            num_units_per_layer=self._hparams.encoder_units_per_layer[1],
                            attended_memory=self._video_encoder.get_data().outputs,
                            attended_memory_length=self._video_data.inputs_length,
                        ) 
                    elif self._hparams.architecture in ('dual_av_align','dual_av_align_tr','dual_av_align_tr_ga', 'dual_av_align_ga'):
                        pass
                    else:
                        raise Exception('Unknown architecture')
            else:
                self._audio_encoder = None

    def _make_decoder(self):

        labels = None
        labels_length = None
        video_len = None
        audio_len = None

        if self._av_encoder is not None:
            video_output = self._av_encoder.get_video_data()
            video_len = self._video_data.inputs_length
            audio_output = self._av_encoder.get_audio_data()
            audio_len = self._audio_data.inputs_length
            labels = self._audio_data.labels
            labels_length = self._audio_data.labels_length
        
        else:    
            if self._video_encoder is not None:
                video_output = self._video_encoder.get_data()
                labels = self._video_data.labels
                labels_length = self._video_data.labels_length
                video_len = self._video_data.inputs_length
            else:
                video_output = None
    
            if self._audio_encoder is not None:
                audio_output = self._audio_encoder.get_data()
                labels = self._audio_data.labels
                labels_length = self._audio_data.labels_length
                audio_len = self._audio_data.inputs_length
            else:
                audio_output = None

        if labels is None or labels_length is None:
            raise Exception('labels are None')

        if self._hparams.architecture in ('unimodal', 'av_align', 'av_align_test'):

            if audio_output is not None:
                output = audio_output
                features_len = audio_len
            else:
                output = video_output
                features_len = video_len

            self._decoder = Seq2SeqUnimodalDecoder(
                encoder_output=output,
                encoder_features_len=features_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams
            )
        elif self._hparams.architecture == 'va_align':
            if video_output is not None:
                output = video_output
                features_len = video_len
            else:
                output = audio_output
                features_len = audio_len
            self._decoder = Seq2SeqUnimodalDecoder(
                    encoder_output=output,
                    encoder_features_len=features_len,
                    labels=labels,
                    labels_length=labels_length,
                    mode=self._mode,
                    hparams=self._hparams,
                    )
        elif self._hparams.architecture == 'va_align_ga':
            if video_output is not None:
                output = video_output
                features_len = video_len
            else:
                output = audio_output
                features_len = audio_len
            self._decoder = Seq2SeqUnimodalDecoder(
                    encoder_output=output,
                    encoder_features_len=features_len,
                    labels=labels,
                    labels_length=labels_length,
                    mode=self._mode,
                    hparams=self._hparams,
                    ga_loss = self._video_encoder.get_guided_attention_loss()
                    )

        elif self._hparams.architecture in ('bimodal', 'dual_av_align', 'av_align_add_vid_dec'):
            self._decoder = Seq2SeqBimodalDecoder(
                video_output=video_output,
                audio_output=audio_output,
                video_features_len=video_len,
                audio_features_len=audio_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams
            )
 
        elif self._hparams.architecture == 'dual_av_align_tr':
            self._decoder = Seq2SeqBimodalDecoder(
                video_output=video_output,
                audio_output=audio_output,
                video_features_len=video_len,
                audio_features_len=audio_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams,
                tr_loss = self._av_encoder.get_transpose_loss()
            )
        
        elif self._hparams.architecture == 'dual_av_align_tr_ga':
            self._decoder = Seq2SeqBimodalDecoder(
                video_output=video_output,
                audio_output=audio_output,
                video_features_len=video_len,
                audio_features_len=audio_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams,
                tr_loss = self._av_encoder.get_transpose_loss(),
                ga_loss = self._av_encoder.get_guided_attention_loss(),
            )

        elif self._hparams.architecture == 'dual_av_align_ga':
            self._decoder = Seq2SeqBimodalDecoder(
                video_output=video_output,
                audio_output=audio_output,
                video_features_len=video_len,
                audio_features_len=audio_len,
                labels=labels,
                labels_length=labels_length,
                mode=self._mode,
                hparams=self._hparams,
                ga_loss = self._av_encoder.get_guided_attention_loss(),
            )
        else:
            raise Exception('Unknown architecture')

        self.train_op = self._decoder.train_op if self._mode == 'train' else None
        self.batch_loss = self._decoder.batch_loss if self._mode == 'train' else None

        #  Here you may add other variables for easy debugging when running sessions
        # self.dbg = self._decoder._basic_decoder_train_outputs if self._mode == 'train' else None

    def extract_results(self):

        pass

    def _init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=1000)
