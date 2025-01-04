#!/usr/bin/env bash

for FILE in docs/pages/*.ipynb ; do
    jupyter nbconvert --to notebook --execute --inplace $FILE
done