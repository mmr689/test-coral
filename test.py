import tflite_runtime.interpreter as tflite

model_path = 'models/best_yolov3n_int8.tflite'
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
print('A1')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('A2')