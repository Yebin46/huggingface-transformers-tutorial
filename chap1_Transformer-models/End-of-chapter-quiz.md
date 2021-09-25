# Chapter 1 Quiz

### 1. Explore the Hub and look for the roberta-large-mnli checkpoint. What task does it perform?
- https://huggingface.co/roberta-large-mnli
- Text classification
- 그 중에서도 두 개의 문장이 논리적으로 연결되어있는지(옳은지) 세 개의 label(contradiction, neutral, entailment)로 구분하는 task이다. natural language inference라고도 불린다.


### 2. What will the following code return?
~~~ python
from transformers import pipeline

ner = pipeline('ner', grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
~~~
- NER이니까 persons, organizations, locations를 나타내는 단어를 return할 것임.
- grouped_entities=True이니까 여러 단어가 하나의 entity를 의미하면 하나로 grouping 될 것임.


### 3. What should replace ... in this code sample?
~~~ python
from transformers import pipeline

filler =  pipeline('fill-mask', model='bert-base-cased')
result = filler('...')
~~~
- This [MASK] has been waiting for you.
- [MASK]가 포함된 문장이어야 함.


### 4. Why will this code fail?
~~~ python
from transformers import pipeline

classifier = pipeline('zero-shot-classification')
result = classifier('This is a course about the Transformers library')
~~~
- candidate_labels=[...] 옵션을 이용해 분류할 labels를 지정해줘야 함.


### 5. What does 'transfer learning' mean?
- second model을 (randomly 초기화 하지 않고) first model의 weight로 초기화 함으로써 사전학습된 모델의 knowledge를 전달하는 것.


### 6. True or False? A language model usually does not need labels for its pretraining.
- True. pretraining은 주로 self-supervised임. (self-supervised: label이 input으로부터 자동으로 만들어지는 것.)


### 7. Select the sentence the best describes the terms 'model,' 'architecture,' and 'weights.'
- 정답) achitecture는 model을 만들기 위한 mathematical functions의 연속이고, weights는 그 functions의 parameters다.
- 오답) model이 건물이면, architecture는 blueprint고 weights는 그 안에 사는 사람들이다 -> 사람이 아니라 건물의 벽돌이나 다른 건물을 이루는 재료여야 성립함.
- 오답) architecture가 model을 만들기 위한 지도이고, weights는 그 지도에 표시된 도시들이다. -> 전체적으로 문제임. 지도는 보통 하나의 현존하는 현실을 나타냄. 근데 주어진 하나의 architecture에 대해 weight는 여러 후보들이 가능함.


### 8. Which of these types of models would you use for completing prompts with generated text?
- prompt로부터 text를 생성하는 것은 decoder model에 적합함.


### 9. Which of those types of models would you use for summarizing texts?
- 문서를 이해하고 문장을 만들어야 하므로 sequence-to-sequence model에 적합함.


### 10. Which of these types of models would you use for classifying text inputs according to certain labels?
- 전체 문장의 표현(representation)을 생성하는 encoder model에 적합함.


### 11. What possible source can the bias observed in a model have?
- 전이학습시(transfer learning) pretrained model의 bias가 fine-tuned model에도 영향을 끼친다.
- model을 train 시켰던 데이터가 편향되어 있는 경우 (모델이 bias를 갖는 가장 명확한 원인 중 하나)
- model을 optimizing하는 metric이 편향되어 있는 경우. 사용자가 선택한 metric에 따라 어떤 재고 없이 맹목적으로 optimize.
