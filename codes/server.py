import os
import numpy as np
from flask import Flask, request, render_template, send_file, jsonify
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
plt.rcParams['figure.dpi'] = 100  # Reducir DPI de las figuras
import matplotlib.pyplot as plt
import io
import base64
import pkg_resources
import tempfile
import gc
from mlxtend.plotting import plot_confusion_matrix
from scipy.stats import gaussian_kde

# Configurar límites de memoria para numpy
np.ones(1, dtype=np.float64).nbytes  # Forzar inicialización de numpy
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (int(1e9), hard))  # Límite de 1GB

from pyROGER import roger
from pyROGER import models

app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Configuraciones de Flask
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Límite de 50MB para uploads
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

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

def process_in_batches(data, model, columns, batch_size=500):
    """Procesar datos en lotes pequeños"""
    n_samples = len(data)
    pred_prob = np.zeros((n_samples, 5))
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = np.column_stack([data[i:end_idx, col] for col in columns])
        pred_prob[i:end_idx] = model.predict_prob(batch, n_model=0)
        gc.collect()
    
    return pred_prob

def create_plot(data, prob, r_col, v_col, titles, cmaps, dpi=100):
    """Crear plot con configuraciones optimizadas"""
    fig, axes = plt.subplots(1, 5, figsize=(10, 2), dpi=dpi)
    plt.subplots_adjust(wspace=0)
    
    for i, ax in enumerate(axes):
        scatter = ax.scatter(data[:, r_col], 
                           data[:, v_col],
                           c=prob[:, i],
                           cmap=cmaps[i],
                           s=20)  # Reducir tamaño de puntos
        ax.set_title(titles[i], fontsize=8)
        ax.tick_params(labelsize=6)
        
    plt.tight_layout()
    
    # Guardar plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route("/")
def index():
    return render_template("index.html", models=AVAILABLE_MODELS)

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files["file"]
        validation_file = request.files.get("validation_file", None)
        
        selected_model = request.form.get('model', 'model1')
        columns = [
            int(request.form.get('r_column', 3)),
            int(request.form.get('v_column', 4)),
            int(request.form.get('m_column', 5))
        ]
        
        v_columns = [
            int(request.form.get('validation_r_column', columns[0])),
            int(request.form.get('validation_v_column', columns[1])),
            int(request.form.get('validation_m_column', columns[2]))
        ]
        real_class_column = int(request.form.get('real_class_column', 0))

        titles = ['$P_{CL}$', '$P_{BS}$', '$P_{RIN}$', '$P_{IN}$', '$P_{ITL}$']
        cmaps = ['Reds', 'Oranges', 'Greens', 'Blues', 'Greys']

        if not file or not file.filename.endswith(('.npy', '.dat')):
            return "Invalid format. Please upload a .npy or .dat file"

        # Procesar archivo principal
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            data = np.load(filepath, mmap_mode='r') if file.filename.endswith('.npy') else np.loadtxt(filepath)
        except Exception as e:
            return f"Error loading file: {str(e)}"
        finally:
            os.remove(filepath)
        
        # Procesar modelo
        if selected_model == 'model1':
            models.HighMassRoger1.train(path_to_saved_model=[
                os.path.join(DATA_PATH, 'HighMassRoger1_KNN.joblib'),
                os.path.join(DATA_PATH, 'HighMassRoger1_RF.joblib'),
                os.path.join(DATA_PATH, 'HighMassRoger1_SVM.joblib')
            ])
            
            pred_prob = process_in_batches(data, models.HighMassRoger1, columns[:2])
            pred_prob[:, [2, 1]] = pred_prob[:, [1, 2]]
        
        elif selected_model == 'model2':
            models.Roger2.train(path_to_saved_model=[os.path.join(DATA_PATH, 'roger2_KNN.joblib')])
            pred_prob = process_in_batches(data, models.Roger2, columns)
        
        # Crear y guardar plot principal
        img_data = create_plot(data, pred_prob, columns[0], columns[1], titles, cmaps)
        
        # Guardar resultados
        result_path = os.path.join(UPLOAD_FOLDER, "resultado.dat")
        np.savetxt(result_path, pred_prob, fmt='%.6f', header="P_CL P_BS P_RIN P_IN P_ITL")
        
        # Variables para validación
        plot2_data = plot_conf_data = plot_density_data = None
        has_validation = False
        
        # Procesar validación si existe
        if validation_file and validation_file.filename.endswith(('.npy', '.dat')):
            has_validation = True
            val_filepath = os.path.join(UPLOAD_FOLDER, validation_file.filename)
            
            try:
                validation_data = (np.load(val_filepath, mmap_mode='r') if validation_file.filename.endswith('.npy') 
                                 else np.loadtxt(val_filepath))
                real_classes = validation_data[:, real_class_column].copy()
                
                # Procesar validación en lotes
                if selected_model == 'model1':
                    val_pred_prob = process_in_batches(validation_data, models.HighMassRoger1, v_columns[:2])
                    val_pred_prob[:, [2, 1]] = val_pred_prob[:, [1, 2]]
                else:
                    val_pred_prob = process_in_batches(validation_data, models.Roger2, v_columns)
                
                # Crear plots de validación con DPI reducido
                plot2_data = create_plot(validation_data, val_pred_prob, v_columns[0], v_columns[1], 
                                       titles, cmaps, dpi=80)
                
                # Crear plot de matriz de confusión
                conf_mat, pred_class = models.HighMassRoger1.confusion_matrix(
                    thresholds=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                    pred_prob=val_pred_prob,
                    real_class=real_classes
                )

                # Guardar matriz de confusión
                conf_mat_path = os.path.join(UPLOAD_FOLDER, "confusion_matrix.dat")
                header = "CL BS RIN IN ITL"
                np.savetxt(conf_mat_path, conf_mat, fmt='%.6f', header=header)

                # Crear plot de matriz de confusión
                fig_conf = plt.figure(figsize=(8, 6))
                plot_confusion_matrix(conf_mat, show_absolute=True, show_normed=True,
                                    class_names=['CL','BS','RIN','IN','ITL'])
                
                # Save confusion matrix plot to buffer
                img_buf_conf = io.BytesIO()
                plt.savefig(img_buf_conf, format='png', bbox_inches='tight', dpi=300)
                img_buf_conf.seek(0)
                plot_conf_data = base64.b64encode(img_buf_conf.read()).decode('utf-8')
                plt.close(fig_conf)

                # Crear plots de densidad por clase
                fig_density = plt.figure(figsize=(15, 3))
                class_names = ['CL', 'BS', 'RIN', 'IN', 'ITL']
                
                # Crear todos los subplots primero
                axes = []
                for i in range(5):
                    ax = plt.subplot(1, 5, i+1)
                    ax.set_title(class_names[i])
                    ax.set_xlabel('R/R200')
                    if i == 0:  # Solo para el primer plot
                        ax.set_ylabel('V/V200')
                    ax.grid(True)
                    axes.append(ax)

                # Establecer límites comunes para todos los plots
                x_max = max(validation_data[:, v_columns[0]])*1.1
                y_min = min(validation_data[:, v_columns[1]])*1.1
                y_max = max(validation_data[:, v_columns[1]])*1.1

                # Aplicar límites a todos los axes
                for ax in axes:
                    ax.set_xlim(0, x_max)
                    ax.set_ylim(y_min, y_max)
                
                # Llenar solo los paneles que tienen datos
                for i, (class_name, cmap) in enumerate(zip(class_names, cmaps)):
                    mask = real_classes == (i+1)
                    if np.any(mask):  # Solo si hay datos para esta clase
                        # Datos para la clase actual
                        x = validation_data[mask, v_columns[0]]
                        y = validation_data[mask, v_columns[1]]
                        
                        # Calcular la densidad
                        xy = np.vstack([x, y])
                        z = gaussian_kde(xy)(xy)
                        
                        # Normalizar z para que esté entre 0 y 1
                        z = (z - z.min()) / (z.max() - z.min())
                        
                        # Ordenar los puntos por densidad
                        idx = z.argsort()
                        x, y, z = x[idx], y[idx], z[idx]
                        
                        # Crear scatter plot usando el mismo colormap que en val_pred_prob
                        scatter = axes[i].scatter(x, y, c=z, s=50, cmap=cmap)

                plt.tight_layout()
                
                # Save density plots to buffer
                img_buf_density = io.BytesIO()
                plt.savefig(img_buf_density, format='png', bbox_inches='tight', dpi=300)
                img_buf_density.seek(0)
                plot_density_data = base64.b64encode(img_buf_density.read()).decode('utf-8')
                plt.close(fig_density)

            finally:
                if os.path.exists(val_filepath):
                    os.remove(val_filepath)
        
        gc.collect()
        
        return render_template("result.html", 
                             plot1=img_data,
                             plot2=plot2_data,
                             plot_conf=plot_conf_data,
                             plot_density=plot_density_data,
                             model_name=AVAILABLE_MODELS[selected_model],
                             has_validation=has_validation)

    except Exception as e:
        gc.collect()
        return f"An error occurred: {str(e)}"

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

@app.route("/download_confusion_matrix")
def download_confusion_matrix():
    result_path = os.path.join(UPLOAD_FOLDER, "confusion_matrix.dat")
    return send_file(result_path, 
                    as_attachment=True, 
                    download_name="confusion_matrix.dat",
                    mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)

