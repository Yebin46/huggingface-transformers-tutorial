# Sequence-to-sequence models
https://huggingface.co/course/chapter1/7?fw=pt

- sequence-to-sequence models는 Transformer model에서 encoder와 decoder 모두 사용함. (그래서 encoder-decoder 모델이라고 불림)
- 각 stage마다 encoder의 attention layers는 initial sentence의 모든 단어에 접근 할 수 있지만 decoder의 attention layers는 이전 시점의 단어들에만 접근할 수 있음.
- encoder가 sentence의 numerical representation을 만들고, 이 representation이 decoder의 input으로 들어감.
- decoder는 word를 생성
- summarization, translation, generative QA처럼 주어진 input에 따라 새로운 문장들을 만드는 task를 잘함.
- encoder와 decoder의 weights를 공유할 필요 없음. -> output과 input의 길이가 달라도 됨.

### sequence-to-sequence model examples
- BART
- mBART
- Marian
- T5
- 이외에도 독립된 encoder와 decoder로부터 seq2seq model을 만들 수도 있음 (e.g. bert & gpt-2)
