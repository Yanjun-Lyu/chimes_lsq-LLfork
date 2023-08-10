#! /bin/bash
# Be sure to set compute system with, e.g., export hosttype="LLNL-LC" before running

DEBUG=${1-0}
VERBOSE=${2-0}
FLAGS="CXX=mpicxx"

echo $hosttype

if [[ "$hosttype" == "UM-ARC" ]] ; then
    echo "here"
    source ../../../modfiles/UM-ARC.mod
    module list
    FLAGS="CXX=mpiicc"
elif [[ "$hosttype" == "LLNL-LC" ]] ; then
    source ../../../modfiles/LLNL-LC.mod
fi



make clean  2>&1 /dev/null

if [[ $DEBUG == '1' ]] ; then
    make debug $FLAGS
else
    make opt $FLAGS
fi
