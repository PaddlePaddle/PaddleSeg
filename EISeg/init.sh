#!/bin/bash

ROOT=`cd "$(dirname ${BASH_SOURCE[0]})" && pwd`

echo "ROOT : $ROOT"

export PYTHONPATH=$PYTHONPATH:$ROOT/eiseg
