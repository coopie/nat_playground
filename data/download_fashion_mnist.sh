#!/bin/sh

git clone https://github.com/zalandoresearch/fashion-mnist.git
mv fashion-mnist/data/fashion data/fashion-mnist
rm -rf fashion-mnist
