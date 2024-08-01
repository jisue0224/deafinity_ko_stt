# deafinity_ko_stt
한국어 음성을 교정 없이 들리는 대로 text로 변환해주는 model

# **개요**

**Training**
1. 한국어 음성 data / 문장 data load
2. 문장 data -> (g2p) -> 발음 나는대로 text data로 변환
3. 음성 및 올바른 발음 data로 모델 학습

**Validation**
1. gpt api로 한국어 문장 생성
2. 해당 문장 직접 음성 녹음 (일부러 틀리게 발음하거나 옳게 발음하거나)
3. 직접 녹음한 음성 파일과 함께 발음 문장을 함께 제공
4. 정확도 측정
