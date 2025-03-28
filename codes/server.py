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
    'model2': 'ROGER_v2',
    'model3': 'ROGER_v3'
}

@app.route("/")
def index():
    return render_template("index.html", models=AVAILABLE_MODELS)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    validation_file = request.files.get("validation_file", None)
    
    selected_model = request.form.get('model', 'model1')
    
    # Columnas para el set de entrenamiento
    r_column = int(request.form.get('r_column', 3))
    v_column = int(request.form.get('v_column', 4))
    m_column = int(request.form.get('m_column', 5))
    
    # Columnas para el set de validación
    v_r_column = int(request.form.get('validation_r_column', r_column))
    v_v_column = int(request.form.get('validation_v_column', v_column))
    v_m_column = int(request.form.get('validation_m_column', m_column))

    # Definir variables para plotting
    titles = ['$P_{CL}$', '$P_{BS}$', '$P_{RIN}$', '$P_{IN}$', '$P_{ITL}$']
    cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Greys']

    if file and (file.filename.endswith(".npy") or file.filename.endswith(".dat")):
        # Procesar archivo principal
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        if file.filename.endswith(".npy"):
            data = np.load(filepath)
        else:
            data = np.loadtxt(filepath)
        
        # Procesar archivo de validación si existe
        validation_data = None
        if validation_file and (validation_file.filename.endswith(".npy") or validation_file.filename.endswith(".dat")):
            val_filepath = os.path.join(UPLOAD_FOLDER, validation_file.filename)
            validation_file.save(val_filepath)
            
            if validation_file.filename.endswith(".npy"):
                validation_data = np.load(val_filepath)
            else:
                validation_data = np.loadtxt(val_filepath)

        # Aplicar el modelo seleccionado
        if selected_model == 'model1':
            # Crear array con las columnas seleccionadas
            selected_data = np.column_stack((data[:, r_column], 
                                       data[:, v_column]))  # Agregamos logM
            models.HighMassRoger1.train(path_to_saved_model = [DATA_PATH + '/HighMassRoger1_KNN.joblib',
                                                               DATA_PATH + '/HighMassRoger1_RF.joblib',
                                                               DATA_PATH + '/HighMassRoger1_SVM.joblib'])
            # Usar ROGER para clusters masivos
            pred_class = models.HighMassRoger1.predict_class(selected_data, n_model=0)
            pred_prob = models.HighMassRoger1.predict_prob(selected_data, n_model=0)
            
            # Intercambiar columnas 2 y 3
            pred_prob[:, [2, 1]] = pred_prob[:, [1, 2]]
            
            plot_title = 'ROGER Analysis - High Mass Clusters'
            
            # Si hay datos de validación, procesarlos también
            if validation_data is not None:
                val_selected_data = np.column_stack((validation_data[:, v_r_column], 
                                                   validation_data[:, v_v_column]))
                val_pred_prob = models.HighMassRoger1.predict_prob(val_selected_data, n_model=0)
                val_pred_prob[:, [2, 1]] = val_pred_prob[:, [1, 2]]
                
                # Crear plot para datos de validación
                fig2, axes2 = plt.subplots(1, 5, figsize=(10, 2), sharey=True, sharex=True)
                plt.subplots_adjust(wspace=0)
                
                for i, ax in enumerate(axes2):
                    scatter = ax.scatter(validation_data[:, v_r_column], 
                                       validation_data[:, v_v_column],
                                       cmap=cmaps[i],
                                       c=val_pred_prob[:,i],
                                       s=50)
                    ax.set_title(titles[i])
                    ax.set_xlabel('R/R200')
                    ax.grid(True)
                    ax.set_xlim(0, max(validation_data[:, v_r_column])*1.1)
                    ax.set_ylim(min(validation_data[:, v_v_column])*1.1, 
                               max(validation_data[:, v_v_column])*1.1)
                    ax.set_ylabel('')
                axes2[0].set_ylabel('V/V200')
                plt.tight_layout()

                # Save validation plot to buffer
                img_buf2 = io.BytesIO()
                plt.savefig(img_buf2, format='png', bbox_inches='tight', dpi=300)
                img_buf2.seek(0)
                plot2_data = base64.b64encode(img_buf2.read()).decode('utf-8')
                plt.close(fig2)

                # Guardar resultados de validación
                val_result_path = os.path.join(UPLOAD_FOLDER, "validation_resultado.dat")
                header = "P_CL P_BS P_RIN P_IN P_ITL"
                np.savetxt(val_result_path, val_pred_prob, fmt='%.6f', header=header)

        elif selected_model == 'model2':
            # Crear array con las columnas seleccionadas
            selected_data = np.column_stack((data[:, r_column], 
                                       data[:, v_column],
                                       data[:, m_column]))  # Agregamos logM
            # Usar ROGER para clusters pequeños
            #roger_model = roger.ROGER(model_type='small_mass')  # Ejemplo
            models.Roger2.train(path_to_saved_model = [DATA_PATH + '/roger2_KNN.joblib'])

            pred_class = models.Roger2.predict_class(selected_data, n_model=0)
            pred_prob = models.Roger2.predict_prob(selected_data, n_model=0)
            plot_title = 'ROGER2 Analysis'
            
        elif selected_model == 'model3':
            # Usar ROGER v2
            roger_model = models.ROGERv2()  # Ejemplo
            resultado = roger_model.predict(data)
            plot_title = 'ROGER v2 Analysis'

        # Create matplotlib plot con 5 paneles
        fig, axes = plt.subplots(1, 5, figsize=(10, 2), sharey=True, sharex=True)
        plt.subplots_adjust(wspace=0) 
        
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
        result_path = os.path.join(UPLOAD_FOLDER, "resultado.dat")
        
        # Guardar las probabilidades en formato .dat
        header = "P_CL P_BS P_RIN P_IN P_ITL"
        np.savetxt(result_path, pred_prob, fmt='%.6f', header=header)

        return render_template("result.html", 
                             plot1=img_data,
                             plot2=plot2_data if validation_data is not None else None,
                             model_name=AVAILABLE_MODELS[selected_model],
                             has_validation=validation_data is not None)

    return "Invalid format. Please upload a .npy or .dat file"

@app.route("/download")
def download_result():
    result_path = os.path.join(UPLOAD_FOLDER, "resultado.dat")
    return send_file(result_path, 
                    as_attachment=True, 
                    download_name="roger_results.dat",
                    mimetype='text/plain')

@app.route("/download_validation")
def download_validation_result():
    result_path = os.path.join(UPLOAD_FOLDER, "validation_resultado.dat")
    return send_file(result_path, 
                    as_attachment=True, 
                    download_name="roger_validation_results.dat",
                    mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)

