#!/bin/bash

cmd=$(./a.out)
while [ "$cmd"=="0 1 2 3" ]
do
  cmd=$(./a.out)
done
echo $cmd
