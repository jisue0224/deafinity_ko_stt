import hgtk  # 한글 자모 분리 및 결합 라이브러리
import torch
import json

# 한글 자모 정의
consonants = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
vowels = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"

# 자모와 기타 기호 정의
symbols = "0123456789.,?! "

# 전체 문자 집합
characters = consonants + vowels + symbols

# 문자 -> 인덱스 변환 딕셔너리
char_to_index = {char: idx for idx, char in enumerate(characters)}

# 인덱스 -> 문자 변환 딕셔너리
index_to_char = {idx: char for idx, char in enumerate(characters)}

# 한글 음절을 자모로 분리하는 함수 정의
def split_to_jamo(text):
    jamo_text = ''
    for char in text:
        if hgtk.checker.is_hangul(char):
            jamo_text += hgtk.text.decompose(char, compose_code=' ')
        else:
            jamo_text += char
    return jamo_text.replace(' ', '')

def join_jamo(jamo_text):
    text = ''
    jamo_list = []
    for char in jamo_text:
        if char in consonants or char in vowels:
            jamo_list.append(char)
        else:
            if jamo_list:
                text += hgtk.text.compose(''.join(jamo_list))
                jamo_list = []
            text += char
    if jamo_list:
        text += hgtk.text.compose(''.join(jamo_list))
    return text

# 라벨을 인덱스로 변환하는 함수 정의
def label_to_index(labels):
    indices = []
    for label in labels:
        jamo_text = split_to_jamo(label)
        indices.append([char_to_index[char] for char in jamo_text])
    return torch.tensor(indices, dtype=torch.long)

# 디코딩 함수 정의
def decode(outputs):
    decoded_batch = []
    for i in range(outputs.size(0)):
        decoded_seq = []
        for t in range(outputs.size(1)):
            # 가장 높은 확률을 가진 클래스 선택
            topv, topi = outputs[i, t].topk(1)
            char_index = topi.item()
            if char_index != 0:  # 0은 CTC의 blank 토큰을 의미
                decoded_seq.append(index_to_char[char_index])
        decoded_text = join_jamo(''.join(decoded_seq))
        decoded_batch.append(decoded_text)
    return decoded_batch

# JSON 파일 로드 함수
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 데이터 저장 함수
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
