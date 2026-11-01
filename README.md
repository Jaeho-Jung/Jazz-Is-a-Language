# Jazz Is a Language

코드 진행과 박자 정보를 바탕으로 재즈 솔로를 생성하는 모델을 실험합니다. `src/`에는 RNN, LSTM, decoder-only Transformer를 NumPy와 PyTorch로 구현한 코드가 있습니다.

## 코드 범위

- `src/RNN_numpy/`, `src/LSTM_numpy/`: NumPy 기반 순환 모델과 학습 코드
- `src/RNN_pytorch/`, `src/LSTM_pytorch/`: PyTorch 기반 순환 모델
- `src/Transformer_numpy/`, `src/Transformer_pytorch/`: decoder-only Transformer 구현

PyTorch Transformer의 기본 설정은 `d_model=128`, attention head 4개, layer 3개, dropout 0.3, label smoothing 0.1입니다. 학습 결과는 이 설정만으로 판단할 수 없습니다.

Encoder-decoder 구조와 상대 위치 주의를 비교한 코드는 별도 [MusicTransformer-Pytorch 저장소](https://github.com/Jaeho-Jung/MusicTransformer-Pytorch)에 있습니다.
