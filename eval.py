import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from g2pk import G2p
from utils import characters, decode, load_json
from model import SpeechRecognitionModel

# 데이터셋 클래스 정의
class SpeechDataset(Dataset):
    def __init__(self, json_data, audio_dir, transform=None):
        self.data = json_data
        self.audio_dir = audio_dir
        self.transform = transform
        self.mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=128)
        self.g2p = G2p()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        audio_path = os.path.join(self.audio_dir, entry["발화정보"]["fileNm"])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Apply SpecAugment if provided
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        pronunciation = entry["발화정보"]["pronunciation"]
        return mel_spec, pronunciation

# STT 결과 출력 함수
def transcribe(model, dataloader):
    model.eval()
    transcriptions = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.permute(0, 2, 1)  # (batch, freq, time) -> (batch, time, freq)
            outputs = model(inputs)
            decoded_output = decode(outputs)  # 모델 출력을 텍스트로 변환
            transcriptions.extend(decoded_output)
    return transcriptions

# 메인 함수
def main():
    # 경로 하드코딩
    eval_metadata_file = 'data/eval_metadata.json'
    eval_audio_dir = 'data/eval_audio_files'
    
    # 데이터 로드
    eval_data = load_json(eval_metadata_file)
    eval_dataset = SpeechDataset(eval_data, eval_audio_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # 하이퍼파라미터 설정
    input_dim = 128  # Mel-spectrogram의 주파수 차원
    hidden_dim = 256
    output_dim = len(characters)  # 문자 집합의 크기
    num_layers = 3

    # 모델 초기화
    model = SpeechRecognitionModel(input_dim, hidden_dim, output_dim, num_layers)
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load('best_model.pt'))

    # STT 결과 출력
    transcriptions = transcribe(model, eval_loader)
    for transcription in transcriptions:
        print(transcription)

if __name__ == "__main__":
    main()
