import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F

# Improved STT 모델 정의
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SpeechRecognitionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# SpecAugment 변환기 정의
class SpecAugment:
    def __init__(self, freq_mask_param=15, time_mask_param=35, num_freq_masks=2, num_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram):
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            spectrogram = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(spectrogram)
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            spectrogram = T.TimeMasking(time_mask_param=self.time_mask_param)(spectrogram)
        
        return spectrogram
