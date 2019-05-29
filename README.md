# TemporalUT3
Temporal difference learning for ultimate tic-tac-toe.

## What is ultimate tic-tac-toe?
It's like tic-tac-toe, but each square of the game contains another game of tic-tac-toe in it! Win small games to claim the squares in the big game. Simple, right? But there is a catch: Whichever small square you pick is the next big square your opponent must play in. [Read more...](https://docs.riddles.io/ultimate-tic-tac-toe/rules)

![ultimate tic-tac-toe gif](https://static-content.riddles.io/ultimate-tic-tac-toe-objectives-small-squares.gif)

## What is temporal difference learning?
Temporal difference (TD) learning is a reinforcement learning algorithm trained only using self-play. The algorithm learns by bootstrapping from the current estimate of the value function, i.e. the value of a state is updated based on the current estimate of the value of future states. [Read more...](https://en.wikipedia.org/wiki/Temporal_difference_learning)

## How to use

```bash
python train.py
```

or set the learning hyperparameters using the following arguments:

```bash
python train.py [--lr LEARN_RATE] [--a ALPHA] [--e EPSILON]
```

## Experiments

Coming soon.

## To-do
 - Write human player class to allow playing against the bot.
 - [Scale the value of terminal results by the game length to prefer shorter games](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68).
 - Implement UT3 neural network in other frameworks, eg: TensorFlow.
 - Make asynchronous, i.e. do self-play, neural net training and model comparison in parallel.

## Requirements
 - [PyTorch](https://pytorch.org/)
 - [Progress](https://pypi.org/project/progress/)

## Thanks
 - [Sam Culley](https://github.com/swculley)
