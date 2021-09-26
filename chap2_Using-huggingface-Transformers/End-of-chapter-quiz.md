# Chapter 2 Quiz

### 1. What is the order of the language modeling pipeline?
- tokenizer가 text를 받아서 IDs로 return하고, 이 IDs를 model이 받아서 predictions을 출력한다. 그럼 tokenizer가 다시 이 predictions를 text로 변환하는 데 사용된다.
- model이 출력한 predictions가 tokenizer 없이 바로 text가 될 수 없음에 유의.

### 2. How many dimensions does the tensor output by the base Transformer model have, and what are they?
- 3: sequence length, batch size, hidden size

### 3. Which of the following is an example of subword tokenization?
- WordPiece(Bert, DistilBERT), BPE(GPT-2, RoBERTa), Unigram(XLNET, ALBERT)
- 오답) Character-based tokenization: subword tokenization은 word-based와 character-based 사이임.
- 오답) Splitting on whitespace and punctuation -> word-based tokenization을 의미함.

### 4. What is a model head?
- 보통 하나 혹은 몇 개의 layer로 이루어진 추가적인 요소. transformer prediction을 특정 task에 맞게 변환해줌.

### 5. What is AutoModel?
- checkpoint에 맞는 architecture를 return하는 객체.
- AutoModel은 올바른 architecture를 return하기 위해 단지 초기화할 checkpoint만 필요함.

### 6. What are the techniques to be aware of when batching sequences of different lengths together?
- Truncating: 길이가 다른 sequence들을 잘라서 직사각형 모양에 맞도록 고르게 만드는 방법
- Padding: 길이가 다른 sequence들에 토큰을 덧붙여서 직사각형 모양에 맞도록 고르게 만드는 방법
- Attention masking: 길이가 다른 sequence들을 처리할 때 중요

### 7. What is the point of applying a SoftMax function to the logits output by a sequence classification model?
- softmax의 결과가 0에서 1사이로 만들어지므로, 상한선과 하한선을 만들어줌으로써 이해할 수 있도록 만들어준다.
- softmax의 결과의 합이 1이 되어서 확률적 해석이 가능해진다.

### 8. What method is most of the tokenizer API centered around?
- tokenizer object를 바로 호출하는 것.
- tokenizer의 __call__ method는 거의 모든 것을 다루는 강력한 method임.

### 9. What does the result variable contain in this code sample?
~~~ python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
result = tokenizer.tokenize('Hello!')
~~~
- 문자열로 이루어진 하나의 list. (이걸 IDs로 바꾸고 model의 인풋으로 넣어줘야 함.)

### 10. Is there something wrong with the following code?
~~~ python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModel.from_pretrained('gpt2')

encoded = tokenizer('Hey!', return_tensors='pt')
result = model(**encoded)
~~~
- tokenizer와 model은 같은 checkpoint를 사용해야 한다.
