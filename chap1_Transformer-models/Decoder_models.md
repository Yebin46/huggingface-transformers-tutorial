# Decoder models
https://huggingface.co/course/chapter1/6?fw=pt

- decoder models는 Transformer model에서 decoder만 사용함.
- auto-regressive model이라고 불림.
- decoder: Uni-directional, Auto-regressive, Masked self-attention + Cross-attention
- 각 stage마다 attention layers는 sentence에 있는 이전 시점의 단어들에만 접근 할 수 있음.
- text generation에 특화.

### Decoder model examples
- CTRL (Conditional Transformer Language Model for Controllable Generation)
- GPT & GPT-2
- Transformer XL
