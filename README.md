# Jazz Is a Language

코드 진행과 박자 정보를 입력해 재즈 솔로를 생성합니다. RNN, LSTM, decoder-only Transformer를 NumPy와 PyTorch로 구현한 코드는 `src/`에 있습니다.

## 구현

- `src/RNN_numpy/`, `src/LSTM_numpy/`: NumPy 기반 순환 모델과 학습 코드
- `src/RNN_pytorch/`, `src/LSTM_pytorch/`: PyTorch 기반 순환 모델
- `src/Transformer_numpy/`, `src/Transformer_pytorch/`: decoder-only Transformer 구현

PyTorch Transformer는 기본값으로 `d_model=128`, attention head 4개, layer 3개, dropout 0.3, label smoothing 0.1을 사용합니다. 이 값은 모델 설정이며 학습 성능을 나타내지 않습니다.

Encoder-decoder 구조에서 상대 위치 주의를 비교한 실험은 [MusicTransformer-Pytorch 저장소](https://github.com/Jaeho-Jung/MusicTransformer-Pytorch)에 따로 정리했습니다.
