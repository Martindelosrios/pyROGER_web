import os
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
import matplotlib.pyplot as plt
import io
import base64
import pkg_resources
import tempfile

from pyROGER import roger
from pyROGER import models

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Configurar el puerto para Render
port = int(os.environ.get('PORT', 5000))

UPLOAD_FOLDER = tempfile.gettempdir()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Obtener la ruta absoluta del DATA_PATH
DATA_PATH = pkg_resources.resource_filename("pyROGER", "../dataset/")
print(f"Initial DATA_PATH: {DATA_PATH}")  # Debug print

# Verificar si el directorio existe
if not os.path.exists(DATA_PATH):
    print(f"DATA_PATH no existe, intentando rutas alternativas")
    # Intentar encontrar la ruta correcta
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "../dataset/"),
        os.path.join(os.path.dirname(__file__), "../../dataset/"),
        pkg_resources.resource_filename("pyROGER", "dataset/"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            DATA_PATH = path
            print(f"Found valid DATA_PATH: {DATA_PATH}")
            break

# Lista de modelos disponibles
AVAILABLE_MODELS = {
    'model1': 'ROGER_v1 (high mass clusters)',
    'model2': 'ROGER_v1 (small mass clusters)',
    'model3': 'ROGER_v2'
}

@app.route("/")
def index():
    return render_template("index.html", models=AVAILABLE_MODELS)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    selected_model = request.form.get('model', 'model1')
    r_column = int(request.form.get('r_column', 3))  # default a columna 3
    v_column = int(request.form.get('v_column', 4))  # default a columna 4

    if file and file.filename.endswith(".npy"):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load and process the file
        data = np.load(filepath)
        
        # Crear array con las columnas seleccionadas
        selected_data = np.column_stack((data[:, r_column], data[:, v_column]))
        
        # Aplicar el modelo seleccionado
        if selected_model == 'model1':
            models.HighMassRoger1.train(path_to_saved_model = [DATA_PATH + '/HighMassRoger1_KNN.joblib',
                                                               DATA_PATH + '/HighMassRoger1_RF.joblib',
                                                               DATA_PATH + '/HighMassRoger1_SVM.joblib'])
            # Usar ROGER para clusters masivos
            pred_class = models.HighMassRoger1.predict_class(selected_data, n_model=0)
            pred_prob = models.HighMassRoger1.predict_prob(selected_data, n_model=0)
            plot_title = 'ROGER Analysis - High Mass Clusters'
            
        elif selected_model == 'model2':
            # Usar ROGER para clusters pequeños
            #roger_model = roger.ROGER(model_type='small_mass')  # Ejemplo
            pred_class = np.round(data[:,0])
            pred_prob = np.random.uniform(size=(len(data),5)) 
            plot_title = 'ROGER Analysis - Small Mass Clusters'
            
        elif selected_model == 'model3':
            # Usar ROGER v2
            roger_model = models.ROGERv2()  # Ejemplo
            resultado = roger_model.predict(data)
            plot_title = 'ROGER v2 Analysis'

        # Create matplotlib plot con 5 paneles
        fig, axes = plt.subplots(1, 5, figsize=(10, 2), sharey=True, sharex=True)

        plt.subplots_adjust(wspace=0) 
        # Definir títulos y probabilidades para cada panel
        titles = ['$P_{CL}$', '$P_{BS}$', '$P_{RIN}$', '$P_{IN}$', '$P_{ITL}$']
        cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Greys']
        
        # Crear los 5 paneles
        for i, ax in enumerate(axes):
            scatter = ax.scatter(data[:, r_column], 
                               data[:, v_column],
                               cmap=cmaps[i],
                               c=pred_prob[:,i],
                               s=50)
            ax.set_title(titles[i])
            ax.set_xlabel('R/R200')
            ax.grid(True)
            
            # Mantener los mismos límites en todos los paneles
            ax.set_xlim(0, max(data[:, r_column])*1.1)
            ax.set_ylim(min(data[:, v_column])*1.1, max(data[:, v_column])*1.1)
            ax.set_ylabel('')
        axes[0].set_ylabel('Vu/V200')
        plt.tight_layout()

        # Save plot to buffer
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        # Save results
        result_path = os.path.join(UPLOAD_FOLDER, "resultado.npy")
        np.savez(result_path, 
                 classes=pred_prob, 
                 probabilities=pred_prob)

        # Obtener listado de archivos en DATA_PATH
        try:
            print(f"Trying to read DATA_PATH: {DATA_PATH}")  # Debug print
            if os.path.exists(DATA_PATH):
                # Listar todos los archivos y directorios
                all_files = []
                for root, dirs, files in os.walk(DATA_PATH):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, DATA_PATH)
                        all_files.append(f"{rel_path} ({full_path})")
                
                if all_files:
                    data_files = all_files
                    print(f"Found {len(data_files)} files in {DATA_PATH}")
                else:
                    data_files = ["Directory is empty"]
                    print(f"No files found in {DATA_PATH}")
            else:
                data_files = [f"DATA_PATH does not exist: {os.path.abspath(DATA_PATH)}"]
                print(f"DATA_PATH not found: {os.path.abspath(DATA_PATH)}")
        except Exception as e:
            data_files = [f"Error reading DATA_PATH ({os.path.abspath(DATA_PATH)}): {str(e)}"]
            print(f"Error reading DATA_PATH: {str(e)}")

        return render_template("result.html", 
                             plot1=img_data,
                             model_name=AVAILABLE_MODELS[selected_model],
                             data_path=os.path.abspath(DATA_PATH),
                             data_files=data_files)

    return "Invalid format. Please upload a .npy file"

@app.route("/download")
def download_result():
    result_path = os.path.join(UPLOAD_FOLDER, "resultado.npy")
    return send_file(result_path, 
                    as_attachment=True, 
                    download_name="roger_results.npz")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)

