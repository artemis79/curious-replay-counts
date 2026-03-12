#!/bin/sh
set -eu

apt-get update
apt-get install -y wget
apt-get install -y unrar
apt-get clean

pip3 install gymnasium 
pip3 install ale-py
pip3 install opencv-python

# mkdir roms && cd roms
# wget -L -nv http://www.atarimania.com/roms/Roms.rar
# unrar x -o+ Roms.rar
# python3 -m atari_py.import_roms ROMS
# cd .. && rm -rf roms

pip3 install autorom[accept-rom-license]
AutoROM --accept-license
