#!/bin/bash
KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python autoencoder_test.py $1 $2 $3
