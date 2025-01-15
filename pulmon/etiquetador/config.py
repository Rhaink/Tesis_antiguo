"""
Configuraciones y constantes del programa de etiquetado.
"""

# Configuración de ventana
VENTANA_NOMBRE = "Radiografia"
VENTANA_ANCHO = 640
VENTANA_ALTO = 640

# Colores (B, G, R)
COLOR_LINEA = (0, 0, 255)        # Rojo
COLOR_PUNTO = (0, 255, 0)        # Verde
COLOR_LINEA_CENTRAL = (255, 0, 0) # Azul

# Teclas
TECLAS = {
    'MOVER_IZQ_3': 'q',
    'MOVER_DER_3': 'w',
    'MOVER_IZQ_4': 'e',
    'MOVER_DER_4': 'r',
    'MOVER_IZQ_5': 'a',
    'MOVER_DER_5': 'd',
    'MOVER_IZQ_6': 'f',
    'MOVER_DER_6': 'g',
    'MOVER_IZQ_7': 'z',
    'MOVER_DER_7': 'c',
    'MOVER_IZQ_8': 'v',
    'MOVER_DER_8': 'b',
    'MOVER_IZQ_12': 'y',
    'MOVER_DER_12': 'u',
    'MOVER_IZQ_13': 'h',
    'MOVER_DER_13': 'j',
    'MOVER_IZQ_14': 'n',
    'MOVER_DER_14': 'm',
    'MOVER_IZQ_15': 'i',
    'MOVER_DER_15': 'o',
    'PREVISUALIZAR': 'p',
    'LIMPIAR': 'l',
    'GUARDAR': 's',
    'SIGUIENTE': 'x',
    'TERMINAR': 't'
}

# Mensajes
MENSAJES = {
    'PRIMER_CLICK': 'Primer Click hecho',
    'SEGUNDO_CLICK': 'Segundo Click hecho',
    'TERCER_CLICK': 'Tercer Click hecho',
    'DEFINIR_COORDENADAS': 'Por favor, define todas las coordenadas antes de previsualizar.',
    'REGISTRO_GUARDADO': 'Registro Guardado'
}

# Texto del menú
MENU_TEXTO = """----------------------------
MENU
Q: Movimiento punto 3  a la izquierda
W: Movimiento punto 3 a la derecha
E: Movimiento punto 4 a la izquierda
R: Movimiento punto 4 a la derecha
A: Movimiento punto 5  a la izquierda
D: Movimiento punto 5 a la derecha
F: Movimiento punto 6 a la izquierda
G: Movimiento punto 6 a la derecha
Z: Movimiento punto 7  a la izquierda
C: Movimiento punto 7 a la derecha
V: Movimiento punto 8 a la izquierda
B: Movimiento punto 8 a la derecha
Y: Movimiento punto 12  a la izquierda
U: Movimiento punto 12 a la derecha
H: Movimiento punto 13 a la izquierda
J: Movimiento punto 13 a la derecha
N: Movimiento punto 14 a la izquierda
M: Movimiento punto 14 a la derecha
I: Movimiento punto 15 a la izquierda
O: Movimiento punto 15 a la derecha
P: Previsualizar
L: Limpiar imagen
S: Guardar información
X: Siguiente Imagen
T: Cerrar Programa
----------------------------"""
