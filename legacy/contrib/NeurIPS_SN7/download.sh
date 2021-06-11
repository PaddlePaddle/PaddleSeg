#!/bin/bash

mkdir -p /wdata/saved_model/hrnet/
wget https://www.dropbox.com/s/krtl5tmrkf4qv56/prefix.tar.gz?dl=1 -O prefix.tar.gz
tar -zxf prefix.tar.gz
cp -r prefix /wdata/saved_model/hrnet/best_model
