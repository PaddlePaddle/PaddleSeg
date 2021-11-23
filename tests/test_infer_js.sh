#!/bin/bash

set -o errexit
set -o nounset

cd tests/web
# run humanseg test in chrome
./node_modules/.bin/jest --config ./jest.config.js