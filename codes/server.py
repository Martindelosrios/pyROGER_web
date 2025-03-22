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

DATA_PATH = pkg_resources.resource_filename("pyROGER", "dataset/")

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
            models.HighMassRoger1.train(path_to_saved_model = [DATA_PATH + 'pyROGER/HighMassRoger1_KNN.joblib',
                                                               DATA_PATH + 'pyROGER/HighMassRoger1_RF.joblib',
                                                               DATA_PATH + 'pyROGER/HighMassRoger1_SVM.joblib'])
            # Usar ROGER para clusters masivos
            pred_class = models.HighMassRoger1.predict_class(selected_data, n_model=0)
            pred_prob = models.HighMassRoger1.predict_prob(selected_data, n_model=0)
            plot_title = 'ROGER Analysis - High Mass Clusters'
            
        elif selected_model == 'model2':
            # Usar ROGER para clusters pequeños
            #roger_model = roger.ROGER(model_type='small_mass')  # Ejemplo
            pred_class = np.round(data[:,0])
            pred_prob = np.random.uniform(size = (len(data), 5))
            plot_title = DATA_PATH#'ROGER Analysis - Small Mass Clusters'
            
        elif selected_model == 'model3':
            # Usar ROGER v2
            roger_model = models.ROGERv2()  # Ejemplo
            resultado = roger_model.predict(data)
            plot_title = 'ROGER v2 Analysis'

        # Create matplotlib plot con 5 paneles
        fig, axes = plt.subplots(1, 5, figsize=(10, 2), sharey=True, sharex=True)

        plt.subplots_adjust(wspace=0) 
        # Definir títulos y probabilidades para cada panel
        titles = [DATA_PATH, '$P_{BS}$', '$P_{RIN}$', '$P_{IN}$', '$P_{ITL}$']
        cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Greys']
        
        # Crear los 5 paneles
        for i, ax in enumerate(axes):
            scatter = ax.scatter(data[:, r_column], 
                               data[:, v_column],
                               cmap=cmaps[i],
                               c=pred_prob[:,i],
                               s=50)
            #ax.set_title(titles[i])
            ax.set_xlabel('R/R200')
            ax.grid(True)
            
            # Mantener los mismos límites en todos los paneles
            ax.set_xlim(0, max(data[:, r_column])*1.1)
            ax.set_ylim(min(data[:, v_column])*1.1, max(data[:, v_column])*1.1)
            ax.set_ylabel('')
        axes[0].set_title(titles[0])
        axes[0].set_ylabel('V/V200')
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

        return render_template("result.html", 
                             plot1=img_data,
                             model_name=AVAILABLE_MODELS[selected_model])

    return "Invalid format. Please upload a .npy file"

@app.route("/download")
def download_result():
    result_path = os.path.join(UPLOAD_FOLDER, "resultado.npy")
    return send_file(result_path, 
                    as_attachment=True, 
                    download_name="roger_results.npz")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)

