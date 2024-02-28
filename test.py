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


# Dibuja los cuadros delimitadores en la imagen original
for detection in output_data[0]:
    score = detection[2]
    print(score)
    if score > 0:  # Filtra detecciones con confianza baja
        box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        box = box.astype(int)
        cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)

# Guarda la imagen con los cuadros delimitadores
cv2.imwrite('imagen_con_detecciones.jpg', image)