# AI Zero to Hero

## Motivation

I wanted to get a basic understanding of AI.

Friday - Wednesday:

[X] Bayes Networks: Exact and Sampling
[X] Markov Chains

Thursday - Friday:

[x] CS188 Neural networks
[X] Multi-armed bandits
[X] Reinforcement learning I and II

Saturday:

[x] Neural network + gradient descent (micrograd)
[x] Makemore - bigram and single layer neural network

Sunday:

Bonus exercises
Exercises:
E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?
E06: meta-exercise! Think of a fun/interesting exercise and complete it.

1. Transformer implementation
