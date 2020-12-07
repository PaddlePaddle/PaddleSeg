#!/bin/bash
set -o errexit

base_path=$(cd `dirname $0`/../..; pwd)
cd $base_path

python dataset/download_pet.py
