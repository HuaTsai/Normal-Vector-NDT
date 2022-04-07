#!/bin/sh
find . -regex '\.\/src\/.*\.\(h\|hpp\|cc\|cpp\)' -exec clang-format -style=file -i {} \;

files=`find . -regex '\.\/src\/.*\.\(h\|hpp\|cc\|cpp\)'`
for file in $files
do
    if [ -n "$(tail -c 1 "$file")" ]
    then
        echo "" >> $file
    fi
done
