import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

model_path = 'models/best_yolov3n_int8.tflite'
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
print('A1')

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('A2')

# Carga una imagen de entrada (ajusta la ruta según tu caso)
image = cv2.imread('imgs/img_20221116_051503.jpg')

# Preprocesamiento de la imagen
input_shape = input_details[0]['shape']
input_data = cv2.resize(image, (input_shape[1], input_shape[2]))
input_data = np.expand_dims(input_data, axis=0)
input_data = (input_data.astype(np.float32) - 127.5) / 127.5

# Realiza la inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Postprocesamiento de los resultados
# (ajusta esta parte según el formato de salida de tu modelo)

# Imprime los resultados
print("Resultados de la inferencia:", output_data)