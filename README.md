Real Robot Challenge 2022 Training Code (Team Name: decimalcurlew)
========================

This is the training code (team name: decimalcurlew) for the real-robot stage of the [Real Robot Challenge
2022](https://real-robot-challenge.com).

Training environments and datasets can be found at [rrc_2022_datasets](https://github.com/rr-learning/rrc_2022_datasets).

For simulation-based evaluation, we reccomend downgrading [gym](https://github.com/openai/gym) to [gym==0.24.1] after installing the rrc_2022_datasets.

Networks are trained using [Tensorflow 2.9.0](https://github.com/tensorflow/tensorflow/tree/2.9.0) and Python 3.8.

Usage
----------------

Experiments on all environemnts can be run by running:

    $ sh run.sh

Experiments on a single environment can be run by calling:

    $ python main.py --env "trifinger-cube-push-real-expert-v0"
    $ python main.py --env "trifinger-cube-push-real-mixed-v0"
    $ python main.py --env "trifinger-cube-lift-real-expert-v0"
    $ python main.py --env "trifinger-cube-lift-real-mixed-v0"

Hyper-parameters can be modified with different arguments to main.py.
