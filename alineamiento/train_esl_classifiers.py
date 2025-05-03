# Tesis/alineamiento/train_esl_classifiers.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Rutas ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Si este script está en alineamiento/
    ALINEAMIENTO_DIR = CURRENT_SCRIPT_DIR if os.path.basename(CURRENT_SCRIPT_DIR) == 'alineamiento' else os.path.dirname(CURRENT_SCRIPT_DIR)
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models") # Directorio para guardar modelos
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")   # Directorio para gráficas
except NameError:
    ALINEAMIENTO_DIR = '.'
    RESULTS_DIR = os.path.join(ALINEAMIENTO_DIR, "results")
    MODELS_DIR = os.path.join(ALINEAMIENTO_DIR, "models")
    PLOTS_DIR = os.path.join(ALINEAMIENTO_DIR, "plots")
    logger.info("Ejecutando en modo interactivo, rutas ajustadas.")

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

LINE_PATCHES_PATH = os.path.join(RESULTS_DIR, 'esl_line_patches_train.npz')
ORIENTATION_PATCHES_PATH = os.path.join(RESULTS_DIR, 'esl_orientation_patches_train.npz')

# --- Parámetros del Modelo y Entrenamiento ---
PATCH_SIZE = 64
INPUT_SHAPE = (PATCH_SIZE, PATCH_SIZE, 1) # H, W, Canales
VALIDATION_SPLIT = 0.20 # 20% para validación
NUM_EPOCHS = 50         # Máximo de épocas (EarlyStopping decidirá)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5 # Épocas a esperar sin mejora antes de parar
# ------------------------------------------

def build_esl_classifier(input_shape, learning_rate=LEARNING_RATE):
    """Construye y compila el modelo CNN para clasificación binaria."""
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Añadir más capas Conv/Pool si es necesario
            layers.Flatten(),
            layers.Dropout(0.5), # Regularización
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"), # Salida binaria
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary(print_fn=logger.info) # Imprimir resumen en logs
    return model

def plot_training_history(history, model_name, output_dir):
    """Guarda gráficas de loss y accuracy del entrenamiento."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfica de Accuracy
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title(f'{model_name} - Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')
    axs[0].grid(True, linestyle='--', alpha=0.6)


    # Gráfica de Loss
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title(f'{model_name} - Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
    try:
        plt.savefig(plot_path)
        logger.info(f"Gráfica de historial guardada en: {plot_path}")
    except Exception as e:
        logger.error(f"Error guardando gráfica: {e}")
    plt.close(fig)


def train_classifier(model_name, patches, labels, input_shape):
    """Entrena un clasificador ESL y guarda el mejor modelo."""
    logger.info(f"--- Iniciando entrenamiento para: {model_name} ---")

    if patches is None or labels is None or len(patches) != len(labels):
        logger.error(f"Datos inválidos para {model_name}. Parches: {patches.shape if patches is not None else 'None'}, Etiquetas: {labels.shape if labels is not None else 'None'}")
        return

    logger.info(f"Datos para {model_name}: Parches shape={patches.shape}, Etiquetas shape={labels.shape}")
    logger.info(f"Distribución de etiquetas: {np.bincount(labels)}")

    # Dividir en entrenamiento y validación
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            patches, labels,
            test_size=VALIDATION_SPLIT,
            random_state=42, # Para reproducibilidad
            stratify=labels # Importante para clases desbalanceadas
        )
        logger.info(f"Dividido en: {len(X_train)} entrenamiento, {len(X_val)} validación.")
    except ValueError as e:
        logger.error(f"Error en train_test_split para {model_name} (quizás por pocas muestras por clase?): {e}")
        logger.error("Verificar distribución de etiquetas y tamaño del dataset.")
        return # No continuar si no se puede dividir

    # Construir el modelo
    model = build_esl_classifier(input_shape)

    # Callbacks
    model_checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_weights_only=False, # Guardar modelo completo
        monitor='val_accuracy',  # Guardar basado en accuracy de validación
        mode='max',
        save_best_only=True,     # Solo guardar el mejor
        verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',      # Monitorear pérdida de validación
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, # Restaurar pesos del mejor epoch al final
        verbose=1
    )

    # Entrenar
    logger.info(f"Iniciando model.fit para {model_name}...")
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback, early_stopping_callback],
            verbose=2 # Mostrar una línea por época
        )
        logger.info(f"Entrenamiento finalizado para {model_name}.")

        # Graficar historial
        plot_training_history(history, model_name, PLOTS_DIR)

        # Guardar el modelo final (que tiene los pesos restaurados por EarlyStopping)
        final_model_path = os.path.join(MODELS_DIR, f"{model_name}_final.h5")
        model.save(final_model_path)
        logger.info(f"Modelo final guardado en: {final_model_path} (pesos corresponden a la mejor 'val_loss')")
        logger.info(f"Mejor modelo (basado en 'val_accuracy') guardado en: {model_checkpoint_path}")


    except Exception as e:
        logger.error(f"Error durante el entrenamiento de {model_name}: {e}", exc_info=True)

    logger.info(f"--- Entrenamiento para {model_name} completado ---")


def main():
    logger.info("====== Iniciando Entrenamiento de Clasificadores ESL ======")

    # --- Entrenar Clasificadores de Líneas ---
    logger.info(f"Cargando datos de parches de líneas desde: {LINE_PATCHES_PATH}")
    if not os.path.exists(LINE_PATCHES_PATH):
        logger.error(f"Archivo no encontrado: {LINE_PATCHES_PATH}. Ejecute esl_patch_extractor.py primero.")
        return
    try:
        line_data = np.load(LINE_PATCHES_PATH)
        data_loaded = True
    except Exception as e:
        logger.error(f"Error cargando archivo de parches de líneas: {e}")
        data_loaded = False
        line_data = {} # Para evitar errores posteriores

    if data_loaded:
        for i in range(1, 5):
            model_name = f"esl_classifier_l{i}"
            patches_key = f"patches_l{i}"
            labels_key = f"labels_l{i}"
            if patches_key in line_data and labels_key in line_data:
                 train_classifier(model_name, line_data[patches_key], line_data[labels_key], INPUT_SHAPE)
            else:
                 logger.error(f"No se encontraron las claves {patches_key} o {labels_key} en {LINE_PATCHES_PATH}")
        line_data.close() # Cerrar archivo

    # --- Entrenar Clasificador de Orientación ---
    logger.info(f"\nCargando datos de parches de orientación desde: {ORIENTATION_PATCHES_PATH}")
    if not os.path.exists(ORIENTATION_PATCHES_PATH):
        logger.error(f"Archivo no encontrado: {ORIENTATION_PATCHES_PATH}. Ejecute esl_orientation_patch_extractor.py primero.")
        return
    try:
        orientation_data = np.load(ORIENTATION_PATCHES_PATH)
        data_loaded = True
    except Exception as e:
        logger.error(f"Error cargando archivo de parches de orientación: {e}")
        data_loaded = False
        orientation_data = {}

    if data_loaded:
        model_name = "esl_classifier_theta"
        patches_key = "patches_theta"
        labels_key = "labels_theta"
        if patches_key in orientation_data and labels_key in orientation_data:
             train_classifier(model_name, orientation_data[patches_key], orientation_data[labels_key], INPUT_SHAPE)
        else:
            logger.error(f"No se encontraron las claves {patches_key} o {labels_key} en {ORIENTATION_PATCHES_PATH}")
        orientation_data.close()

    logger.info("====== Entrenamiento de Clasificadores ESL Finalizado ======")


if __name__ == "__main__":
    # Opcional: Configurar GPU si está disponible
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configurar para usar memoria dinámicamente (evita acaparar toda la VRAM)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs disponibles.")
        except RuntimeError as e:
            # La configuración de memoria debe hacerse al inicio
            logger.error(f"Error configurando GPU: {e}")
    else:
        logger.info("No se detectaron GPUs. Usando CPU.")

    main()