#!/bin/bash

for N in 512 1024 2048 4096 8192 16384 32768
do
	echo "Running for m,n,k=$N"
	./gemmEx --m $N --n $N --k $N --n-runs 10
done

for N in 512 1024 2048 4096 8192 16384 32768
do
	echo "Running for m,n,k=$N"
	./gemmEx --m $N --n $N --k $N --n-runs 10 --copy
done