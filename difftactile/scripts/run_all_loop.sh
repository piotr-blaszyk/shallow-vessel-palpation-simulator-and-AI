#!/usr/bin/env zsh
set -e
set -x

for i in {1..10}; do
  ./difftactile/scripts/run_all.sh
done


