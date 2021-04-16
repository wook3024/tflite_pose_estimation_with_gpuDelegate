#!/usr/bin/env bash

cd "$(dirname "$(readlink -f "$0")")"

URL="https://github.com/am15h/tflite_flutter_plugin/releases/download/"
TAG="v0.2.0"

ANDROID_DIR="android/app/src/main/jniLibs/"
ANDROID_LIB="libtensorflowlite_c.so"

ARM_DELEGATE="libtensorflowlite_c_arm_delegate.so"
ARM_64_DELEGATE="libtensorflowlite_c_arm64_delegate.so"
ARM="libtensorflowlite_c_arm.so"
ARM_64="libtensorflowlite_c_arm64.so"
X86="libtensorflowlite_c_x86.so"
X86_64="libtensorflowlite_c_x86_64.so"

POSE_ESTIMATION_URL="https://tfhub.dev/tensorflow/lite-model/posenet/mobilenet/float/075/1/default/1?lite-format=tflite"
POSE_ESTIMATION_MODEL="posenet_mobilenet_float_075_1_default_1.tflite"

delegate=0

while getopts "d" OPTION
do
	case $OPTION in
		d)  delegate=1;;
	esac
done


download () {
    wget "${URL}${TAG}/$1"
    mkdir -p "${ANDROID_DIR}$2/"
    mv $1 "${ANDROID_DIR}$2/${ANDROID_LIB}"
}

wget -O "posenet_mobilenet_float_075_1_default_1.tflite" "${POSE_ESTIMATION_URL}"
mv "${POSE_ESTIMATION_MODEL}" "assets/${POSE_ESTIMATION_MODEL}"

if [ ${delegate} -eq 1 ]
then

download ${ARM_DELEGATE} "armeabi-v7a"
download ${ARM_64_DELEGATE} "arm64-v8a"

else

download ${ARM} "armeabi-v7a"
download ${ARM_64} "arm64-v8a"

fi

download ${X86} "x86"
download ${X86_64} "x86_64"
