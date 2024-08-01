import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from jiwer import wer
from g2pk import G2p
from utils import characters, label_to_index, decode, load_json, save_json
from model import SpeechRecognitionModel, SpecAugment

# 발음 데이터 생성 함수
def generate_pronunciation(data):
    g2p = G2p()
    for entry in data:
        original_text = entry["발화정보"]["stt"]
        pronunciation_text = g2p(original_text)
        entry["발화정보"]["pronunciation"] = pronunciation_text
    return data

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

# Character Error Rate (CER) 평가 함수
def evaluate(model, dataloader):
    model.eval()
    cer = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.permute(0, 2, 1)  # (batch, freq, time) -> (batch, time, freq)
            outputs = model(inputs)
            decoded_output = decode(outputs)  # 모델 출력을 텍스트로 변환
            cer += wer(labels, decoded_output)
    return cer / len(dataloader)

# 학습 코드
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, model_path='best_model.pt'):
    best_cer = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # 입력 데이터를 모델에 맞게 전처리
            inputs = inputs.permute(0, 2, 1)  # (batch, freq, time) -> (batch, time, freq)
            labels = label_to_index(labels)  # 텍스트 라벨을 인덱스로 변환
            
            # CTC 손실 계산을 위한 입력 길이와 타겟 길이
            input_lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
            
            # 모델 출력
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, labels, input_lengths, target_lengths)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 훈련 세트에서 모델 평가
        train_cer = evaluate(model, train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training CER: {train_cer:.4f}')
        
        # 가장 좋은 모델 저장
        if train_cer < best_cer:
            best_cer = train_cer
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with CER: {best_cer:.4f}')

# 메인 함수
def main():
    # 경로 하드코딩
    train_metadata_file = 'data/train_metadata.json'
    train_audio_dir = 'data/train_audio_files'
    
    # 데이터 로드
    train_data = load_json(train_metadata_file)

    # 발음 데이터 생성
    processed_data = generate_pronunciation(train_data)

    # 처리된 데이터 저장
    save_json(processed_data, train_metadata_file)
    print(f'Processed data saved to {train_metadata_file}')

    spec_augment = SpecAugment()
    train_dataset = SpeechDataset(train_data, train_audio_dir, transform=spec_augment)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 하이퍼파라미터 설정
    input_dim = 128  # Mel-spectrogram의 주파수 차원
    hidden_dim = 256
    output_dim = len(characters)  # 문자 집합의 크기
    num_layers = 3

    # 모델, 손실 함수, 옵티마이저 초기화
    model = SpeechRecognitionModel(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, model_path='best_model.pt')

if __name__ == "__main__":
    main()
