#!/bin/bash

for i in `find . -name A.txt`;
do 
	dir=`dirname $i`
	
	cd $dir
	
	echo "Breaking up files in $dir"
	
	# Break into <50M chunks for git push (reconstruct with cat in run_test_suite.sh)
	split -b49M A.txt A.txt.
	rm -f A.txt

	cd - > /dev/null 2>& 1
done
