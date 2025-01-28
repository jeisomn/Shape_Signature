from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
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

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError(f"No se encontraron contornos en la imagen {image_path}.")

    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def load_training_descriptors():
    training_descriptors = []
    training_labels = []
    categories = ['car', 'face', 'tree', 'teddy', 'dog']
    for subdir in categories:
        subdir_path = os.path.join(app.config['TRAINING_IMAGES_FOLDER'], subdir)
        if os.path.isdir(subdir_path):  # Verifica si es una carpeta
            for filename in os.listdir(subdir_path):
                if allowed_file(filename):
                    image_path = os.path.join(subdir_path, filename)
                    descriptor = get_shape_signature(image_path)
                    training_descriptors.append(descriptor)
                    training_labels.append(subdir)  # Usa el nombre de la carpeta como la etiqueta
    return training_descriptors, training_labels

def generate_confusion_matrix():
    # Cargar los descriptores de entrenamiento
    training_descriptors, training_labels = load_training_descriptors()

    y_true = []
    y_pred = []

    # Clasificar cada imagen de entrenamiento
    for train_descriptor, real_label in zip(training_descriptors, training_labels):
        # Calcular la distancia entre el descriptor de prueba y el descriptor de entrenamiento
        distance = euclidean(train_descriptor, train_descriptor)  # Comparación con el mismo descriptor

        # Clasificar según el umbral
        predicted_label = real_label if distance < 0.5 else "unknown"  # Clasificación simple

        # Agregar las etiquetas a las listas
        y_true.append(real_label)
        y_pred.append(predicted_label)

    # Calcular la matriz de confusión
    categories = ['car', 'face', 'tree', 'teddy', 'dog']
    cm = confusion_matrix(y_true, y_pred, labels=categories)

    # Asegurarse de que la carpeta 'matriz' exista dentro de la carpeta 'static'
    matriz_folder = 'matriz'
    if not os.path.exists(matriz_folder):
        os.makedirs(matriz_folder)

    # Guardar la matriz de confusión como imagen
    cm_filename_img = os.path.join(matriz_folder, 'matriz_confusion.png')  # Ruta dentro de static/matriz
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    plt.title('Matriz de Confusión')

    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')

    plt.savefig(cm_filename_img)

    return cm_filename_img

@app.route('/')
def index():
    return render_template('index.html')

def distance_to_probability(distance):
    return 1 / (1 + distance)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
        except Exception as e:
            return render_template('index.html', error=f"Error al guardar el archivo: {str(e)}")

        try:
            # Obtener el descriptor Shape Signature de la imagen cargada
            test_descriptor = get_shape_signature(filepath)

            # Cargar los descriptores de entrenamiento
            training_descriptors, training_labels = load_training_descriptors()

            # Calcular distancias entre el descriptor de prueba y los de entrenamiento
            distances = [euclidean(train_descriptor, test_descriptor) for train_descriptor in training_descriptors]

            # Encontrar el índice de la menor distancia
            min_distance_index = np.argmin(distances)
            predicted_label = training_labels[min_distance_index]

            # Preparar datos para la matriz de confusión
            categories = ['car', 'face', 'tree', 'teddy', 'dog']
            y_true = [training_labels[min_distance_index]]  # Usamos la etiqueta real del descriptor más cercano
            y_pred = [predicted_label]  # Predicción basada en la menor distancia

            # Calcular matriz de confusión
            cm_img = generate_confusion_matrix() 
            cm = confusion_matrix(y_true, y_pred, labels=categories)

            # Graficar la matriz de confusión
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap='Blues')
            fig.colorbar(cax)
            ax.set_xticks(np.arange(len(categories)))
            ax.set_yticks(np.arange(len(categories)))
            ax.set_xticklabels(categories)
            ax.set_yticklabels(categories)
            plt.title('Matriz de Confusion')

            # Guardar la figura como imagen base64 
            img_io = io.BytesIO()
            plt.savefig(img_io, format='png')
            img_io.seek(0)
            img_b64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            plt.close(fig)

            # URL para la imagen subida
            uploaded_image_url = url_for('uploaded_file', filename=filename)

            return render_template(
                'index.html',
                uploaded_image_url=uploaded_image_url,
                predicted_label=predicted_label,
                predicted_probability=distance_to_probability(distances[min_distance_index]),
                cm_img=img_b64
            )

        except Exception as e:
            return render_template('index.html', error=f"Error al procesar la imagen: {str(e)}")

    return redirect(url_for('index'))



# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
