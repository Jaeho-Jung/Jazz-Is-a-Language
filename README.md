# Jazz Is a Language

재즈 솔로의 시퀀스 모델링을 실험하는 프로젝트입니다. RNN, LSTM, decoder-only Transformer의 NumPy 및 PyTorch 구현이 `src/`에 있습니다. 코드 진행과 박자 등 구조화된 음악 피처를 입력으로 사용합니다.

## 코드 범위

- `src/RNN_numpy/`, `src/LSTM_numpy/`: NumPy 기반 순환 모델과 학습 코드
- `src/RNN_pytorch/`, `src/LSTM_pytorch/`: PyTorch 기반 순환 모델
- `src/Transformer_numpy/`, `src/Transformer_pytorch/`: decoder-only Transformer 구현

PyTorch Transformer 설정 파일의 기본값은 `d_model=128`, 4개 attention head, 3개 layer, dropout 0.3, label smoothing 0.1입니다. 이는 코드의 기본 설정이며, 특정 학습 실행의 결과나 성능 수치를 뜻하지 않습니다.

이 저장소는 [MusicTransformer-Pytorch](https://github.com/Jaeho-Jung/MusicTransformer-Pytorch)의 encoder-decoder 및 상대 위치 주의 비교 실험과 별개의 구현입니다. 데이터 원본 솔로 수, 증강 후 샘플 수와 학습에 실제 사용된 수는 실행 기록을 확인한 뒤 명시하겠습니다.
