from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Configura Matplotlib para usar el backend 'Agg'
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['TRAINING_IMAGES_FOLDER'] = '/home/jeison/Escritorio/VisionCom/VisionCom/Practica4/app/static/Dataset'

# Asegúrate de crear la carpeta 'uploads' si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Función para comprobar las extensiones de archivo
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Función para obtener el descriptor Shape Signature (Momentos de Hu)
def get_shape_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"La imagen no se pudo cargar correctamente: {image_path}")
        
    # Usar un umbral adaptativo con Otsu para mejor binarización
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Mostrar la imagen binarizada (solo para depuración)
    # cv2.imshow("Thresholded Image", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError(f"No se encontraron contornos en la imagen {image_path}.")

    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


# Función para clasificar las imágenes de prueba
def classify_image(train_descriptor, test_descriptor):
    # Probar con distancia euclidiana si la similitud del coseno no funciona bien
    distance = euclidean(train_descriptor, test_descriptor)
    return distance

def load_training_descriptors():
    training_descriptors = []
    training_labels = []
    categories = ['car', 'face', 'tree', 'teddy', 'dog']  # Las categorías que mencionaste
    for subdir in categories:
        subdir_path = os.path.join(app.config['TRAINING_IMAGES_FOLDER'], subdir)
        if os.path.isdir(subdir_path):  # Verifica si es una carpeta
            print(f"Procesando la carpeta: {subdir}")  # Mensaje de depuración
            for filename in os.listdir(subdir_path):
                print(f"Procesando archivo: {filename}")  # Mensaje de depuración
                if allowed_file(filename):
                    image_path = os.path.join(subdir_path, filename)
                    descriptor = get_shape_signature(image_path)
                    training_descriptors.append(descriptor)
                    training_labels.append(subdir)  # Usa el nombre de la carpeta como la etiqueta
    return training_descriptors, training_labels


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Obtener el descriptor Shape Signature de la imagen cargada
            test_descriptor = get_shape_signature(filepath)
            print("Test descriptor:", test_descriptor)  # Imprime el descriptor para ver qué contiene

            # Cargar los descriptores de entrenamiento
            training_descriptors, training_labels = load_training_descriptors()

            # Comparar la imagen de prueba con los descriptores de entrenamiento
            distances = [classify_image(train_descriptor, test_descriptor) for train_descriptor in training_descriptors]

            # Clasificar la imagen de prueba basándonos en la menor distancia
            min_distance_index = np.argmin(distances)
            predicted_label = training_labels[min_distance_index]

            # Obtener la etiqueta correcta para la imagen cargada
            true_label = filename.split('-')[0]
            y_true = [true_label]
            y_pred = [predicted_label]

            print("True label:", y_true)  # Imprime la etiqueta verdadera
            print("Predicted label:", y_pred)  # Imprime la etiqueta predicha

            # Calcular la precisión y la matriz de confusión con las 5 categorías
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=['car', 'face', 'tree', 'teddy', 'dog'])

            # Graficar la matriz de confusión
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap='Blues')
            fig.colorbar(cax)
            categories = ['car', 'face', 'tree', 'teddy', 'dog']
            ax.set_xticks(np.arange(len(categories)))
            ax.set_yticks(np.arange(len(categories)))
            ax.set_xticklabels(categories)
            ax.set_yticklabels(categories)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Guardar la figura como imagen base64 para mostrarla en la web
            img_io = io.BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            plt.close(fig)

            return render_template('index.html', accuracy=accuracy, cm_img=img_b64, filename=filename, predicted_label=predicted_label)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return redirect(url_for('index'))


# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
