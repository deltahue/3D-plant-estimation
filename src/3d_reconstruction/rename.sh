#!/bin/bash

NEWFILE=$1

cd $NEWFILE

ls -v | cat -n | 
while read n f; 
do 
    mv -n "$f" "$n.jpg"; 
done
