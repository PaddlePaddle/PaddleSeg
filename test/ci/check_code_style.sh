#!/bin/bash
function abort(){
    echo "Your change doesn't follow PaddleHub's code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

trap 'abort' 0
set -e

cd $TRAVIS_BUILD_DIR
export PATH=/usr/bin:$PATH
pre-commit install

if ! pre-commit run -a ; then
    git diff
    exit 1
fi

trap : 0
