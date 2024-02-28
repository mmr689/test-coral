import tflite_runtime.interpreter as tflite

model_path = 'models/best_yolov3n_int8.tflite'
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])