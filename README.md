# Recurrent Neural Network for Text Calssification
Tensorflow implementation of RNN(Recurrent Neural Network) for sentiment analysis, one of the text classification problems. There are three types of RNN models, 1) Vanilla RNN, 2) Long Short-Term Memory RNN and 3) Gated Recurrent Unit RNN.

![rnn](https://user-images.githubusercontent.com/15166794/39031786-370d0cae-44a5-11e8-8440-27102312274c.png)


## Data: Movie Review
* Movie reviews with one sentence per review. Classification involves detecting positive/negative reviews ([Pang and Lee, 2005](#reference))
* Download "*sentence polarity dataset v1.0*" at the [Official Download Page](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
* Located in *<U>"data/rt-polaritydata/"</U>* in my repository
* *rt-polarity.pos* contains 5331 positive snippets
* *rt-polarity.neg* contains 5331 negative snippets


## Usage
### Train
* positive data is located in *<U>"data/rt-polaritydata/rt-polarity.pos"*</U>
* negative data is located in *<U>"data/rt-polaritydata/rt-polarity.neg"*</U>
* "[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model
* Display help message:

	```bash
	$ python train.py --help
	```

* **Train Example:**

	#### 1. Vanilla RNN
	![vanilla](https://user-images.githubusercontent.com/15166794/39033685-30859e24-44ae-11e8-9d7d-860c75efe080.png)
	
	```bash
	$ python train.py --cell_type "vanilla" \
	--pos_dir "data/rt-polaritydata/rt-polarity.pos" \
	--neg_dir "data/rt-polaritydata/rt-polarity.neg"\
	--word2vec "GoogleNews-vectors-negative300.bin"
	```
		
	#### 2. Long Short-Term Memory (LSTM) RNN
	![lstm](https://user-images.githubusercontent.com/15166794/39033684-3053546e-44ae-11e8-893a-7fa685039ce2.png)
	
	```bash
	$ python train.py --cell_type "lstm" \
	--pos_dir "data/rt-polaritydata/rt-polarity.pos" \
	--neg_dir "data/rt-polaritydata/rt-polarity.neg"\
	--word2vec "GoogleNews-vectors-negative300.bin"
	```
	
	#### 3. Gated Reccurrent Unit (GRU) RNN
	![gru](https://user-images.githubusercontent.com/15166794/39033683-3020ce04-44ae-11e8-821f-1a9652ff5025.png)
	
	```bash
	$ python train.py --cell_type "gru" \
	--pos_dir "data/rt-polaritydata/rt-polarity.pos" \
	--neg_dir "data/rt-polaritydata/rt-polarity.neg"\
	--word2vec "GoogleNews-vectors-negative300.bin"
	```


### Evalutation
* Movie Review dataset has **no test data**.
* If you want to evaluate, you should make test dataset from train data or do cross validation. However, cross validation is not implemented in my project.
* The bellow example just use full rt-polarity dataset same the train dataset
* **Evaluation Example:**

	```bash
	$ python eval.py \
	--pos_dir "data/rt-polaritydata/rt-polarity.pos" \
	--neg_dir "data/rt-polaritydata/rt-polarity.neg" \
	--checkpoint_dir "runs/1523902663/checkpoints"
	```


## Reference
* **Seeing stars: Exploiting class relationships for sentiment categorization with
respect to rating scales** (ACL 2005), B Pong et al. [[paper]](http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.pdf)
* **Long short-term memory** (Neural Computation 1997), J Schmidhuber et al. [[paper]](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
* **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation** (EMNLP 2014), K Cho et al. [[paper]](https://arxiv.org/abs/1406.1078)
* Understanding LSTM Networks [[blog]](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* RECURRENT NEURAL NETWORKS (RNN) – PART 2: TEXT CLASSIFICATION [[blog]](https://theneuralperspective.com/2016/10/06/recurrent-neural-networks-rnn-part-2-text-classification/)


