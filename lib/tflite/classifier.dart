import 'dart:math';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/tflite/recognition.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'stats.dart';

/// Classifier
class Classifier {
  /// Instance of Interpreter
  Interpreter _interpreter;

  /// Labels file loaded as list
  List<String> _labels;

  static const String MODEL_FILE_NAME = "model_gpu.tflite";
  static const String LABEL_FILE_NAME = "labelmap.txt";


  /// Input size of image (height = width = 300)
  static const int INPUT_SIZE = 300;

  /// Result score threshold
  static const double THRESHOLD = 0.5;

  /// [ImageProcessor] used to pre-process the image
  ImageProcessor imageProcessor;

  /// Padding the image to transform into square
  int padSize;


  List<int> _inputShape;
  List<int> _outputShape;
  TfLiteType _outputType = TfLiteType.uint8;

  TensorImage _inputImage;
  TensorBuffer _outputBuffer;
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(127.5, 127.5);
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  var _probabilityProcessor;




  // _interpreterOptions = InterpreterOptions();
  //
  // if (numThreads != null) {
  // _interpreterOptions.threads = numThreads;
  // }


  /// Shapes of output tensors
  List<List<int>> _outputShapes;

  /// Types of output tensors
  List<TfLiteType> _outputTypes;

  /// Number of results to show
  static const int NUM_RESULTS = 10;

  Classifier({
    Interpreter interpreter,
    List<String> labels,
  }) {
    loadModel(interpreter: interpreter);
    loadLabels(labels: labels);
  }

  /// Loads interpreter from asset
  void loadModel({Interpreter interpreter}) async {
    try {
      final gpuDelegateV2 = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            false,
            TfLiteGpuInferenceUsage.fastSingleAnswer,
            TfLiteGpuInferencePriority.minLatency,
            TfLiteGpuInferencePriority.auto,
            TfLiteGpuInferencePriority.auto,
          ));

      var interpreterOptions = InterpreterOptions()..addDelegate(gpuDelegateV2);
      // var interpreterOptions = InterpreterOptions()..threads = 4;


      _interpreter = interpreter ??
          await Interpreter.fromAsset(MODEL_FILE_NAME, options: interpreterOptions);

      _inputShape = _interpreter.getInputTensor(0).shape;
      _outputShape = _interpreter.getOutputTensor(0).shape;
      _outputType = _interpreter.getOutputTensor(0).type;

      _outputBuffer = TensorBuffer.createFixedSize(_outputShape, _outputType);
      _probabilityProcessor =
          TensorProcessorBuilder().add(postProcessNormalizeOp).build();
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  /// Loads labels from assets
  void loadLabels({List<String> labels}) async {
    try {
      _labels =
          labels ?? await FileUtil.loadLabels("assets/" + LABEL_FILE_NAME);
    } catch (e) {
      print("Error while loading labels: $e");
    }
  }


  TensorImage _preProcess() {
    int cropSize = min(_inputImage.height, _inputImage.width);
    return ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(cropSize, cropSize))
        .add(ResizeOp(
        _inputShape[1], _inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR))
        .add(preProcessNormalizeOp)
        .build()
        .process(_inputImage);
  }

  /// Pre-process the image
  TensorImage getProcessedImage(TensorImage inputImage) {
    padSize = max(inputImage.height, inputImage.width);
    if (imageProcessor == null) {
      imageProcessor = ImageProcessorBuilder()
          .add(ResizeWithCropOrPadOp(padSize, padSize))
          .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeMethod.BILINEAR))
          .build();
    }
    inputImage = imageProcessor.process(inputImage);
    return inputImage;
  }

  /// Runs object detection on the input image
  Map<String, dynamic> predict(imageLib.Image image) {
    var predictStartTime = DateTime.now().millisecondsSinceEpoch;

    if (interpreter == null) {
      throw StateError('Cannot run inference, Intrepreter is null');
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    // Create TensorImage from image
    _inputImage = TensorImage.fromImage(image);

    // Pre-process TensorImage
    // inputImage = getProcessedImage(inputImage);
    _inputImage = _preProcess();
    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;

    // TensorBuffers for output tensors
    // TensorBuffer outputLocations = TensorBufferFloat(_outputShapes[0]);
    // TensorBuffer outputClasses = TensorBufferFloat(_outputShapes[1]);
    // TensorBuffer outputScores = TensorBufferFloat(_outputShapes[2]);
    // TensorBuffer numLocations = TensorBufferFloat(_outputShapes[3]);

    // Inputs object for runForMultipleInputs
    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    // List<Object> inputs = [inputImage.buffer];

    // Outputs map
    // Map<int, Object> outputs = {
    //   0: outputLocations.buffer,
    //   1: outputClasses.buffer,
    //   2: outputScores.buffer,
    //   3: numLocations.buffer,
    // };
    // List<Object> outputs = List(1*22*22*34).reshape([1,22,22,34]);

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    // run inference
    // print("inputs");
    // print(image.data.reshape([1,160,160,3]).shape);
    // print(image.data[0].runtimeType);
    // print(_inputShape);
    // print(_outputShape);
    // print(_outputBuffer);
    _interpreter.run(_inputImage.buffer, _outputBuffer.getBuffer());

    var inferenceTimeElapsed =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;

    // Maximum number of results to show
    // int resultsCount = min(NUM_RESULTS, numLocations.getIntValue(0));

    // Using labelOffset = 1 as ??? at index 0
    // int labelOffset = 1;

    // Using bounding box utils for easy conversion of tensorbuffer to List<Rect>
    // List<Rect> locations = BoundingBoxUtils.convert(
    //   tensor: outputLocations,
    //   valueIndex: [1, 0, 3, 2],
    //   boundingBoxAxis: 2,
    //   boundingBoxType: BoundingBoxType.BOUNDARIES,
    //   coordinateType: CoordinateType.RATIO,
    //   height: INPUT_SIZE,
    //   width: INPUT_SIZE,
    // );

    List<Recognition> recognitions = [];

    // for (int i = 0; i < resultsCount; i++) {
    //   // Prediction score
    //   var score = outputScores.getDoubleValue(i);
    //
    //   // Label string
    //   var labelIndex = outputClasses.getIntValue(i) + labelOffset;
    //   var label = _labels.elementAt(labelIndex);
    //
    //   // if (score > THRESHOLD) {
    //   //   // inverse of rect
    //   //   // [locations] corresponds to the image size 300 X 300
    //   //   // inverseTransformRect transforms it our [inputImage]
    //   //   Rect transformedRect = imageProcessor.inverseTransformRect(
    //   //       locations[i], image.height, image.width);
    //   //
    //   //   recognitions.add(
    //   //     Recognition(i, label, score, transformedRect),
    //   //   );
    //   // }
    // }

    var predictElapsedTime =
        DateTime.now().millisecondsSinceEpoch - predictStartTime;

    // print("predictElapsedTime: $predictElapsedTime");
    // print("inferenceTimeElapsed: $inferenceTimeElapsed");
    // print("preProcessElapsedTime: $preProcessElapsedTime");
    // print(_probabilityProcessor.process(_outputBuffer.getIntList()));
    print(_outputBuffer.getIntList().shape);
    print(_outputShape);
    return {
      "recognitions": recognitions,
      "stats": Stats(
          totalPredictTime: predictElapsedTime,
          inferenceTime: inferenceTimeElapsed,
          preProcessingTime: preProcessElapsedTime)
    };
  }

  /// Gets the interpreter instance
  Interpreter get interpreter => _interpreter;

  /// Gets the loaded labels
  List<String> get labels => _labels;
}
