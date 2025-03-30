import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
# from cotlet.phon import phonemize
# from g2p_en import G2p
import librosa 

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# from text_utils import TextCleaner
np.random.seed(1)
random.seed(1)
# DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 512
}
MEL_PARAMS = {
    "n_mels": 128,
    "sample_rate":44_100,
    "n_fft": 2048,
    "win_length": 2048,
    "hop_length": 512
}

    
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” ' 
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
_additions = f"ー()-~_+=0123456789[]<>/%&*#@◌" + chr(860) + chr(861) + chr(862) + chr(863) + chr(864) + chr(865) + chr(866)  
# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_additions)



dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=44100,
                 scaling_factor=1.0  # Add scaling_factor parameter
                ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(sample_rate=44_100, 
                                                              n_mels=128,
                                                              n_fft=2048,
                                                              win_length=2048,
                                                              hop_length=512)
        self.mean, self.std = -4, 4
        
        # Add the beta-binomial interpolator
        self.beta_binomial_interpolator = BetaBinomialInterpolator(scaling_factor=scaling_factor)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        wave, text_tensor, speaker_id = self._load_tensor(data)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_melspec(wave_tensor)

        if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
                mode='linear').squeeze(0)

        acoustic_feature = (torch.log(1e-5 + mel_tensor) - self.mean)/self.std

        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # Generate attention prior matrix
        text_len = text_tensor.size(0)
        mel_len = acoustic_feature.size(1)
        attn_prior = torch.from_numpy(self.beta_binomial_interpolator(mel_len, text_len)).float()

        return wave_tensor, acoustic_feature, text_tensor, attn_prior, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 44100:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=44100)
            
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

# Now modify the Collater class to handle the attention prior
class Collater(object):
    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        
        # Add tensor for attention priors
        attn_priors = torch.zeros((batch_size, max_mel_length, max_text_length)).float()
        
        paths = ['' for _ in range(batch_size)]
        
        for bid, (_, mel, text, attn_prior, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            
            # Handle attention prior
            attn_priors[bid, :mel_size, :text_size] = attn_prior
            
            paths[bid] = path
            assert(text_size < (mel_size//2))

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, attn_priors, paths, waves

        return texts, input_lengths, mels, output_lengths, attn_priors

# Update the build_dataloader function to use the new MelDataset and Collater
def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
