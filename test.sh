#!/bin/sh
# latest=snapshot/mobilenet_iter_73000.caffemodel
# latest=$(ls -t snapshot/*.caffemodel | head -n 1)
latest=snapshot/mobilenet_iter_120000.caffemodel

if test -z $latest; then
    exit 1
fi
echo "I am using $latest"

../../build/tools/caffe train -solver="solver_test.prototxt" \
--weights=$latest \
-gpu 0
