#!/bin/bash

cd $1

wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-devkit.tar.gz
tar -xzvf decathlon-1.0-devkit.tar.gz
rm decathlon-1.0-devkit.tar.gz

cd decathlon-1.0
cd data
wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz
tar -xzvf decathlon-1.0-data.tar.gz
rm decathlon-1.0-data.tar.gz
for filename in *.tar
do
  tar -xvf $filename
  rm $filename
done
