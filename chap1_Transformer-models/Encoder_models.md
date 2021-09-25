# Encoder models
https://huggingface.co/course/chapter1/5?fw=pt

- encoder models는 Transformer model에서 encoder만 사용함.
- encoder: Bi-directional, self-attention
- 각 stage마다 attention layers는 input(initial) sequence의 모든 단어에 접근 할 수 있음.
- 의미있는 정보를 잘 추출해냄. -> 그래서 sequence classification, QA, masked language modeling을 잘함.

### Encoder model examples
- ALBERT
- BERT
- DistilBERT
- ELECTRA
- RoBERTa
