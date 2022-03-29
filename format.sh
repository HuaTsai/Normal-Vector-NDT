#!/bin/sh
find . -regex '\.\/src\/.*\.\(h\|hpp\|cc\|cpp\)' -exec clang-format -style=file -i {} \;
