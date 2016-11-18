#!/bin/bash
KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python cnn.py $1 $2
