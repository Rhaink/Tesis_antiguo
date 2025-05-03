# Tesis/alineamiento/train_mashdl_classifiers.py
# MODIFICADO para ajustar INPUT_DIM a 144 puntos/parches
# Versión con arquitectura SdAE + DNN

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg') # Usar backend no interactivo
import matplotlib.pyplot as plt
import logging
import time

# --- Ajuste de Rutas para Importación ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ALINEAMIENTO_DIR = CURRENT_SCRIPT_DIR # Asume que está en 'alineamiento'
    if ALINEAMIENTO_DIR not in sys.path: sys.path.append(ALINEAMIENTO_DIR)
    BASE_PROJECT_DIR = os.path.dirname(ALINEAMIENTO_DIR)
    if BASE_PROJECT_DIR not in sys.path: sys.path.append(BASE_PROJECT_DIR)
    SRC_DIR_PATH = os.path.join(ALINEAMIENTO_DIR, "src")
    if SRC_DIR_PATH not in sys.path: sys.path.append(SRC_DIR_PATH)
except NameError:
    pass # Ajustar manualmente si es necesario

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
try:
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models")
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
except NameError:
    RESULTS_DIR = "results"; MODELS_DIR = "models"; PLOTS_DIR = "plots"
    logger.info("Ejecutando en modo interactivo, rutas ajustadas a directorios locales.")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Ruta Archivo de Entrada ---
# Usar el archivo generado por el extractor modificado
MASHSDL_DATA_PATH = os.path.join(RESULTS_DIR, 'mashdl_training_hypotheses_144lmk.npz')

# --- Parámetros Globales ---
NUM_MODES = 4
Q_PATCH_SIZE = 21             # Debe coincidir con el extractor
NUM_TOTAL_SHAPE_POINTS = 144  # Número de puntos/parches usados
# ***** CAMBIO CRÍTICO: Actualizar INPUT_DIM *****
INPUT_DIM = NUM_TOTAL_SHAPE_POINTS * (Q_PATCH_SIZE ** 2) # 144 * 441 = 63504
# ***********************************************
NUM_CLASSES = 7               # Número de bins para b_k
VALIDATION_SPLIT = 0.20

# --- Hiperparámetros de Entrenamiento (MANTENEMOS LOS MISMOS POR AHORA) ---
# (Ajustar arquitectura y LR será el PASO 2 si esto no funciona)
PRETRAIN_EPOCHS = 50
PRETRAIN_BATCH_SIZE = 8
PRETRAIN_LR = 0.001
NOISE_STDDEV = 0.1
FINETUNE_EPOCHS_PHASE1 = 20
FINETUNE_EPOCHS_PHASE2 = 80
FINETUNE_BATCH_SIZE = 8
FINETUNE_LR_PHASE1 = 0.0001
FINETUNE_LR_PHASE2 = 0.00005
EARLY_STOPPING_PATIENCE = 25

# --- Arquitectura (MANTENEMOS LA MISMA POR AHORA) ---
# (Ajustar dimensiones será el PASO 2)
ENCODING_DIM1 = 1024
ENCODING_DIM2 = 512
DNN_DIM1 = 512
# DNN_DIM2 = 256 # Simplificado en versión anterior, mantener así por ahora

# ===========================================================
# Definición de Modelos (Sin cambios en las funciones build_*)
# ===========================================================
def build_denoising_autoencoder(input_dim, encoding_dim1, encoding_dim2, noise_stddev):
    # (Tu código existente de build_denoising_autoencoder)
    input_layer = layers.Input(shape=(input_dim,), name='dae_input')
    noisy_input = layers.GaussianNoise(noise_stddev, name='gaussian_noise')(input_layer)
    encoded = layers.Dense(encoding_dim1, activation='relu', name='enc_dense1')(noisy_input)
    encoded = layers.BatchNormalization(name='enc_bn1')(encoded)
    encoded = layers.Dense(encoding_dim2, activation='relu', name='enc_dense2')(encoded)
    encoder_output = layers.BatchNormalization(name='encoder_output_bn')(encoded)
    decoded = layers.Dense(encoding_dim1, activation='relu', name='dec_dense1')(encoder_output)
    decoded = layers.BatchNormalization(name='dec_bn1')(decoded)
    reconstruction_output = layers.Dense(input_dim, activation='linear', name='reconstruction_output')(decoded)
    autoencoder = models.Model(input_layer, reconstruction_output, name='denoising_autoencoder')
    return autoencoder

def build_encoder(autoencoder_model, encoding_dim2):
    # (Tu código existente de build_encoder)
    try:
        encoder_output_layer = autoencoder_model.get_layer('encoder_output_bn')
        if encoder_output_layer is None: raise ValueError
        encoder = models.Model(inputs=autoencoder_model.input, outputs=encoder_output_layer.output, name='encoder')
        return encoder
    except Exception as e:
        logger.error(f"Error al construir el modelo encoder: {e}")
        # Intentar encontrar por índice o dimensión como fallback (menos robusto)
        try:
            # Asumiendo BN después de la última Dense del encoder
            # Busca la capa Dense con encoding_dim2 y toma la salida de la capa BN siguiente
            dense_layer_idx = -1
            for idx, layer in enumerate(autoencoder_model.layers):
                if isinstance(layer, layers.Dense) and layer.units == encoding_dim2:
                    # Verificar si la siguiente capa es BN
                    if idx + 1 < len(autoencoder_model.layers) and isinstance(autoencoder_model.layers[idx+1], layers.BatchNormalization):
                        dense_layer_idx = idx + 1
                        break
            if dense_layer_idx != -1:
                 logger.warning("Encoder construido usando búsqueda por dimensión/tipo (fallback).")
                 encoder = models.Model(inputs=autoencoder_model.input, outputs=autoencoder_model.layers[dense_layer_idx].output, name='encoder_fallback')
                 return encoder
            else:
                 logger.error("Fallback para construir encoder también falló.")
                 return None
        except Exception as fallback_e:
             logger.error(f"Error en fallback de build_encoder: {fallback_e}")
             return None


# Mantener la versión simplificada que usaste antes (512 -> Softmax)
def build_combined_classifier(encoder_model, dnn_dim1, num_classes):
    # (Tu código existente de build_combined_classifier - versión simplificada)
    encoder_model.trainable = False
    dnn_input = encoder_model.output
    # Solo una capa oculta densa antes de la salida
    dnn = layers.Dense(dnn_dim1, activation='relu', name='dnn_dense1')(dnn_input) # e.g., 512
    dnn = layers.BatchNormalization(name='dnn_bn1')(dnn)
    dnn = layers.Dropout(0.5, name='dnn_dropout1')(dnn)
    output = layers.Dense(num_classes, activation='softmax', name='dnn_output')(dnn)
    combined_model = models.Model(inputs=encoder_model.input, outputs=output, name='combined_classifier_simplified')
    return combined_model

# ===========================================================
# Función para Graficar Historial (Sin cambios)
# ===========================================================
def plot_training_history(history, model_name, output_dir):
    # (Tu código existente para plot_training_history)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        axs[0].plot(history.history['accuracy'], label='Train Accuracy')
        axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axs[0].set_title(f'{model_name} - Accuracy'); axs[0].set_ylabel('Accuracy'); axs[0].set_xlabel('Epoch'); axs[0].legend(loc='lower right'); axs[0].grid(True, linestyle='--', alpha=0.6)
    else: axs[0].set_title(f'{model_name} - Accuracy (No data)')
    if 'loss' in history.history and 'val_loss' in history.history:
        axs[1].plot(history.history['loss'], label='Train Loss')
        axs[1].plot(history.history['val_loss'], label='Validation Loss')
        axs[1].set_title(f'{model_name} - Loss'); axs[1].set_ylabel('Loss'); axs[1].set_xlabel('Epoch'); axs[1].legend(loc='upper right'); axs[1].grid(True, linestyle='--', alpha=0.6)
    else: axs[1].set_title(f'{model_name} - Loss (No data)')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
    try: plt.savefig(plot_path); logger.info(f"Gráfica guardada: {plot_path}")
    except Exception as e: logger.error(f"Error guardando gráfica {model_name}: {e}")
    plt.close(fig)

# ===========================================================
# Función Principal de Entrenamiento (Sin cambios en la lógica, solo usa nuevo INPUT_DIM)
# ===========================================================
def train_all_mashdl_modes():
    """Carga los datos y entrena un clasificador SdAE+DNN para cada modo K."""
    logger.info(f"====== Iniciando Entrenamiento MaShDL (INPUT_DIM = {INPUT_DIM}) ======")

    # --- Cargar Datos Generados ---
    logger.info(f"Cargando datos desde: {MASHSDL_DATA_PATH}")
    if not os.path.exists(MASHSDL_DATA_PATH):
        logger.error(f"Archivo no encontrado: {MASHSDL_DATA_PATH}. Ejecute el extractor modificado primero."); return
    try: mashdl_training_data = np.load(MASHSDL_DATA_PATH)
    except Exception as e: logger.error(f"Error cargando datos MaShDL: {e}"); return

    # --- Bucle de Entrenamiento por Modo ---
    for k in range(NUM_MODES):
        model_base_name = f"mashdl_k{k}_144lmk" # Añadir sufijo al nombre del modelo
        features_key = f"features_k{k}"; labels_key = f"labels_k{k}"
        logger.info(f"\n{'='*20} Iniciando Proceso para Modo {k} {'='*20}")

        if features_key not in mashdl_training_data or labels_key not in mashdl_training_data:
            logger.error(f"No se encontraron datos ('{features_key}'/'{labels_key}') modo {k}. Saltando."); continue

        X = mashdl_training_data[features_key].astype(np.float32)
        y = mashdl_training_data[labels_key]

        # Validar dimensiones AHORA con el nuevo INPUT_DIM
        if X.shape[0] == 0 or X.shape[0] != y.shape[0] or X.shape[1] != INPUT_DIM:
             logger.error(f"Datos inválidos/inconsistentes modo {k} ({X.shape}, {y.shape}). Esperado features={INPUT_DIM}. Saltando.")
             continue

        logger.info(f"Datos cargados Modo {k}: Features {X.shape}, Labels {y.shape}")

        # --- Preprocesamiento y División (Sin cambios) ---
        try:
            y_one_hot = utils.to_categorical(y, num_classes=NUM_CLASSES)
            X_train_ae, X_val_ae = train_test_split(X, test_size=VALIDATION_SPLIT, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=VALIDATION_SPLIT, random_state=42, stratify=y)
            logger.info(f"Datos divididos: {X_train.shape[0]} train, {X_val.shape[0]} val.")
        except Exception as e: logger.error(f"Error preprocesando/dividiendo datos modo {k}: {e}. Saltando."); continue

        # --- FASE 1: Pre-entrenamiento (Sin cambios en lógica) ---
        logger.info(f"--- [Modo {k}] Iniciando Pre-entrenamiento SdAE ---")
        start_pretrain_time = time.time()
        autoencoder = build_denoising_autoencoder(INPUT_DIM, ENCODING_DIM1, ENCODING_DIM2, NOISE_STDDEV)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=PRETRAIN_LR), loss='mean_squared_error')
        encoder_model_path = os.path.join(MODELS_DIR, f"{model_base_name}_encoder_pretrained.h5")
        try:
            logger.info("Entrenando Autoencoder...")
            history_ae = autoencoder.fit(X_train_ae, X_train_ae, epochs=PRETRAIN_EPOCHS, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, validation_data=(X_val_ae, X_val_ae), verbose=2)
            encoder = build_encoder(autoencoder, ENCODING_DIM2)
            if encoder:
                 encoder.save(encoder_model_path)
                 logger.info(f"Encoder pre-entrenado guardado: {encoder_model_path}")
                 plot_training_history(history_ae, f"{model_base_name}_DAE_pretrain", PLOTS_DIR)
            else: logger.error("Fallo al construir/guardar encoder. Abortando fine-tuning."); continue
        except Exception as e: logger.error(f"Error pre-entrenamiento Modo {k}: {e}", exc_info=True); continue
        pretrain_duration = time.time() - start_pretrain_time
        logger.info(f"Pre-entrenamiento Modo {k} completado en {pretrain_duration:.2f} seg.")

        # --- FASE 2: Ajuste Fino (Sin cambios en lógica) ---
        logger.info(f"--- [Modo {k}] Iniciando Ajuste Fino (Fine-Tuning) ---")
        start_finetune_time = time.time()
        try:
            logger.info(f"Cargando encoder desde {encoder_model_path}")
            encoder_loaded = models.load_model(encoder_model_path, compile=False)
            combined_model = build_combined_classifier(encoder_loaded, DNN_DIM1, NUM_CLASSES) # Usa la versión simplificada

            # Callbacks
            finetune_checkpoint_path = os.path.join(MODELS_DIR, f"{model_base_name}_SdAE_DNN_best.h5")
            ft_checkpoint = callbacks.ModelCheckpoint(finetune_checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
            ft_early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
            ft_callbacks = [ft_checkpoint, ft_early_stop]

            # Fase 1 FT
            logger.info("Compilando y Entrenando Fase 1 FT (Encoder Congelado)...")
            optimizer_ft1 = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR_PHASE1)
            combined_model.compile(loss="categorical_crossentropy", optimizer=optimizer_ft1, metrics=["accuracy"])
            # combined_model.summary(print_fn=logger.info) # Opcional imprimir resumen aquí
            history_ft1 = combined_model.fit(X_train, y_train, batch_size=FINETUNE_BATCH_SIZE, epochs=FINETUNE_EPOCHS_PHASE1, validation_data=(X_val, y_val), callbacks=ft_callbacks, verbose=2)

            # Fase 2 FT
            logger.info("Descongelando Encoder y recompilando para Fase 2 FT...")
            encoder_loaded.trainable = True
            optimizer_ft2 = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR_PHASE2)
            combined_model.compile(loss="categorical_crossentropy", optimizer=optimizer_ft2, metrics=["accuracy"])
            # combined_model.summary(print_fn=logger.info) # Opcional imprimir resumen aquí
            logger.info(f"Entrenando Fase 2 FT (hasta {FINETUNE_EPOCHS_PHASE1 + FINETUNE_EPOCHS_PHASE2} épocas totales)...")
            history_ft2 = combined_model.fit(X_train, y_train, batch_size=FINETUNE_BATCH_SIZE, epochs=FINETUNE_EPOCHS_PHASE1 + FINETUNE_EPOCHS_PHASE2, initial_epoch=history_ft1.epoch[-1] + 1, validation_data=(X_val, y_val), callbacks=ft_callbacks, verbose=2)

            plot_training_history(history_ft2, f"{model_base_name}_SdAE_DNN_finetune", PLOTS_DIR)
            logger.info(f"Modelo final (mejor val_accuracy) guardado en: {finetune_checkpoint_path}")

        except Exception as e: logger.error(f"Error ajuste fino Modo {k}: {e}", exc_info=True)
        finetune_duration = time.time() - start_finetune_time
        logger.info(f"Ajuste Fino Modo {k} completado en {finetune_duration:.2f} seg.")

    # --- Fin Bucle Modos ---
    try: mashdl_training_data.close()
    except NameError: pass
    logger.info(f"====== Entrenamiento MaShDL (INPUT_DIM = {INPUT_DIM}) Finalizado ======")

# ===========================================================
# Punto de Entrada
# ===========================================================
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"{len(gpus)} Physical GPUs configuradas con memory growth.")
        except Exception as e: logger.error(f"Error configurando GPU: {e}")
    else: logger.info("No se detectaron GPUs. Usando CPU.")

    train_all_mashdl_modes()