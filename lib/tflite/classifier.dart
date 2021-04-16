import 'dart:math';

import 'package:image/image.dart' as imageLib;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'stats.dart';


/// Classifier
class Classifier {
  /// Instance of Interpreter
  Interpreter _interpreter;

  /// Labels file loaded as list
  List<String> _labels;

  static const String MODEL_FILE_NAME = "posenet_mobilenet_float_075_1_default_1.tflite";

  /// Input size of image (height = width = 367)
  static const int INPUT_SIZE = 367;

  /// Result score threshold
  static const double THRESHOLD = 0.5;

  /// [ImageProcessor] used to pre-process the image
  ImageProcessor imageProcessor;

  List<int> _inputShape;
  var _outputTensors;

  TensorImage _inputImage;
  NormalizeOp get preProcessNormalizeOp => NormalizeOp(127.5, 127.5);
  NormalizeOp get postProcessNormalizeOp => NormalizeOp(0, 1);

  Map<int, Object> outputs = {};
  List<Object> inputs = [];

  TensorBuffer float_heatmaps = TensorBufferFloat([1,1,1,1]);
  TensorBuffer float_short_offsets = TensorBufferFloat([1,1,1,1]);
  TensorBuffer float_mid_offsets = TensorBufferFloat([1,1,1,1]);
  TensorBuffer float_segments = TensorBufferFloat([1,1,1,1]);

  /// Shapes of output tensors
  List<List<int>> _outputShapes;

  /// Types of output tensors
  List<TfLiteType> _outputTypes;


  Classifier({
    Interpreter interpreter,
    List<String> labels,
  }) {
    loadModel(interpreter: interpreter);
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
      _outputTensors = _interpreter.getOutputTensors();
      _outputShapes = [];
      _outputTypes = [];
      _outputTensors.forEach((tensor) {
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
      });

      float_heatmaps = TensorBufferFloat(_outputShapes[0]);
      float_short_offsets = TensorBufferFloat(_outputShapes[1]);
      float_mid_offsets = TensorBufferFloat(_outputShapes[2]);
      float_segments = TensorBufferFloat(_outputShapes[3]);

      outputs = {
        0: float_heatmaps.buffer,
        1: float_short_offsets.buffer,
        2: float_mid_offsets.buffer,
        3: float_segments.buffer,
      };
    } catch (e) {
      print("Error while creating interpreter: $e");
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
    _inputImage = _preProcess();

    // Inputs object for runForMultipleInputs
    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    inputs = [_inputImage.buffer];

    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;
    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    // run inference
    // print("inputs");
    // print(image.data.reshape([1,160,160,3]).shape);
    // print(image.data[0].runtimeType);
    // print(_inputShape);
    // print(_outputShape);
    // print(outputs);

    // outputs = [float_heatmaps, float_short_offsets, float_mid_offsets, float_segments]
    _interpreter.runForMultipleInputs(inputs, outputs);

    var inferenceTimeElapsed =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;
    var predictElapsedTime =
        DateTime.now().millisecondsSinceEpoch - predictStartTime;
    
    return {
      "recognitions": float_segments.getIntList(),
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
