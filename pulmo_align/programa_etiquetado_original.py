import cv2
import numpy as np
import pandas as pd
import csv
import math
import sys
import matplotlib.pyplot as plt
import math

def todas_coordenadas_definidas():
    coordenadas = [coord1, coord2, coord3, coord4, coord5, coord6, 
                   coord7, coord8, coord9, coord10, coord11, coord12, coord13, coord14, coord15]
    return all(coord is not None and len(coord) == 2 for coord in coordenadas)

def resize_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def aplicar_clahe(imagen):
    LAB = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(LAB)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl = clahe.apply(L)
    limg = cv2.merge((cl, A, B))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def calcular_grosor(imagen):
    altura, ancho = imagen.shape[:2]
    return max(1, min(altura, ancho) // 200)

def dibujar(event,x,y,flags,param):
    global coord1,coord2,coord3,coord4,coord5,coord6,coord7,coord8,coord9,coord10,coord11,coord12,coord13,coord14,coord15,start,end,medio,tercer,tercer2,cuarto1,cuarto2,cuarto3,quinto1,quinto2,quinto3,quinto4,cuarto1,cuarto2,cuarto3,cuarto4,cuarto5,inicio, final, pendiente,largo,ancho,alfa,grosor_linea,radio_punto

    if event == cv2.EVENT_LBUTTONDOWN:
       coord1 = [int(x), int(y)]
       cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
       print('--------------')
       print('Primer Click hecho')
       print('coordenadas:',coord1)
    
    if event == cv2.EVENT_MOUSEWHEEL:       
        coord2 = [int(x), int(y)]
        cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
        cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
        
        print('--------------')
        print('Segundo Click hecho')
        print('coordenadas:',coord2)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        if coord1[0]!=coord2[0]:
            inicio = coord1
            final = coord2
            medio = [int((coord1[0] + coord2[0])/2), int((coord1[1] + coord2[1])/2)]
            tercer = [int(coord1[0] + (coord2[0] - coord1[0]) / 3), int(coord1[1] + (coord2[1] - coord1[1]) / 3)]
            tercer2 = [int(coord1[0] + 2 * (coord2[0] - coord1[0]) / 3), int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 3)]
            cuarto1 = [int(coord1[0] + (coord2[0] - coord1[0]) / 4), int(coord1[1] + (coord2[1] - coord1[1]) / 4)]
            cuarto2 = [int(coord1[0] + 2 * (coord2[0] - coord1[0]) / 4), int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 4)]
            cuarto3 = [int(coord1[0] + 3 * (coord2[0] - coord1[0]) / 4), int(coord1[1] + 3 * (coord2[1] - coord1[1]) / 4)]
            quinto1 = [int(coord1[0] + (coord2[0] - coord1[0]) / 5), int(coord1[1] + (coord2[1] - coord1[1]) / 5)]
            quinto2 = [int(coord1[0] + 2 * (coord2[0] - coord1[0]) / 5), int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 5)]
            quinto3 = [int(coord1[0] + 3 * (coord2[0] - coord1[0]) / 5), int(coord1[1] + 3 * (coord2[1] - coord1[1]) / 5)]
            quinto4 = [int(coord1[0] + 4 * (coord2[0] - coord1[0]) / 5), int(coord1[1] + 4 * (coord2[1] - coord1[1]) / 5)]
            sexto1 = [int(coord1[0] + (coord2[0] - coord1[0]) / 6), int(coord1[1] + (coord2[1] - coord1[1]) / 6)]
            sexto2 = [int(coord1[0] + 2 * (coord2[0] - coord1[0]) / 6), int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 6)]
            sexto3 = [int(coord1[0] + 3 * (coord2[0] - coord1[0]) / 6), int(coord1[1] + 3 * (coord2[1] - coord1[1]) / 6)]
            sexto4 = [int(coord1[0] + 4 * (coord2[0] - coord1[0]) / 6), int(coord1[1] + 4 * (coord2[1] - coord1[1]) / 6)]
            sexto5 = [int(coord1[0] + 5 * (coord2[0] - coord1[0]) / 6), int(coord1[1] + 5 * (coord2[1] - coord1[1]) / 6)]
            pendiente = (coord1[1] - coord2[1])/(coord1[0] - coord2[0])
            alfa = math.atan(pendiente)
            pendiente = -1/pendiente #pendiente perpendicular 
            coord3 = [int(cuarto1[0]-100), int(y)]
            coord4 = [int(cuarto1[0]+100), int(y)]
            coord5 = [int(cuarto2[0]-100), int(y)]
            coord6 = [int(cuarto2[0]+100), int(y)]
            coord7 = [int(cuarto3[0]-100), int(y)]
            coord8 = [int(cuarto3[0]+100), int(y)]
            coord9 = [int(cuarto1[0]), int(y)]
            coord10 = [int(cuarto2[0]), int(y)]
            coord11 = [int(cuarto3[0]), int(y)]
            coord12 = [int(inicio[0]-80), int(y)]
            coord13 = [int(inicio[0]+80), int(y)]
            coord14 = [int(final[0]-100), int(y)]
            coord15 = [int(final[0]+100), int(y)]
            coord3[1] = int(pendiente*(coord3[0]-cuarto1[0])+cuarto1[1])
            coord4[1] = int(pendiente*(coord4[0]-cuarto1[0])+cuarto1[1])
            coord5[1] = int(pendiente*(coord5[0]-cuarto2[0])+cuarto2[1])
            coord6[1] = int(pendiente*(coord6[0]-cuarto2[0])+cuarto2[1])
            coord7[1] = int(pendiente*(coord7[0]-cuarto3[0])+cuarto3[1])
            coord8[1] = int(pendiente*(coord8[0]-cuarto3[0])+cuarto3[1])
            coord9[1] = int(pendiente*(coord9[0]-cuarto1[0])+cuarto1[1])
            coord10[1] = int(pendiente*(coord10[0]-cuarto2[0])+cuarto2[1])
            coord11[1] = int(pendiente*(coord11[0]-cuarto3[0])+cuarto3[1])
            coord12[1] = int(pendiente*(coord12[0]-inicio[0])+inicio[1])
            coord13[1] = int(pendiente*(coord13[0]-inicio[0])+inicio[1])
            coord14[1] = int(pendiente*(coord14[0]-final[0])+final[1])
            coord15[1] = int(pendiente*(coord15[0]-final[0])+final[1])
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            print('--------------')
            print('Tercer Click hecho')
        if coord1[0]==coord2[0]:
            inicio = coord1
            final = coord2
            medio = [int((coord1[0] + coord2[0])/2), int((coord1[1] + coord2[1])/2)]
            tercer = [coord1[0], int(coord1[1] + (coord2[1] - coord1[1]) / 3)]
            tercer2 = [coord1[0], int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 3)]
            cuarto1 = [coord1[0], int(coord1[1] + (coord2[1] - coord1[1]) / 4)]
            cuarto2 = [coord1[0], int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 4)]
            cuarto3 = [coord1[0], int(coord1[1] + 3 * (coord2[1] - coord1[1]) / 4)]
            quinto1 = [coord1[0], int(coord1[1] + (coord2[1] - coord1[1]) / 5)]
            quinto2 = [coord1[0], int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 5)]
            quinto3 = [coord1[0], int(coord1[1] + 3 * (coord2[1] - coord1[1]) / 5)]
            quinto4 = [coord1[0], int(coord1[1] + 4 * (coord2[1] - coord1[1]) / 5)]
            sexto1 = [coord1[0], int(coord1[1] + (coord2[1] - coord1[1]) / 6)]
            sexto2 = [coord1[0], int(coord1[1] + 2 * (coord2[1] - coord1[1]) / 6)]
            sexto3 = [coord1[0], int(coord1[1] + 3 * (coord2[1] - coord1[1]) / 6)]
            sexto4 = [coord1[0], int(coord1[1] + 4 * (coord2[1] - coord1[1]) / 6)]
            sexto5 = [coord1[0], int(coord1[1] + 5 * (coord2[1] - coord1[1]) / 6)]
            coord3 = [16, int(cuarto1[1])]
            coord4 = [48, int(cuarto1[1])]
            coord5 = [0, int(cuarto2[1])]
            coord6 = [64, int(cuarto2[1])]
            coord7 = [0, int(cuarto3[1])]
            coord8 = [64, int(cuarto3[1])]
            coord9 = [0, int(cuarto1[1])]
            coord10 = [64, int(cuarto2[1])]
            coord11 = [0, int(cuarto3[1])]
            coord12 = [64, int(inicio[1])]
            coord13 = [0, int(inicio[1])]
            coord14 = [64, int(final[1])]
            coord15 = [64, int(final[1])]
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            print('--------------')
            print('Tercer Click hecho')
        cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
        cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

def registrar(nombre_archivo, indice, cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, cx5, cy5, cx6, cy6, cx7, cy7, cx8, cy8, cx9, cy9, cx10, cy10, cx11, cy11, cx12, cy12, cx13, cy13, cx14, cy14, cx15, cy15, imagen):
    with open(nombre_archivo, 'a', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv, delimiter=',')
        writer.writerow([indice, cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, cx5, cy5, cx6, cy6, cx7, cy7, cx8, cy8, cx9, cy9, cx10, cy10, cx11, cy11, cx12, cy12, cx13, cy13, cx14, cy14, cx15, cy15, imagen])

def menu():
    print('----------------------------')
    print('MENU')
    print('Q: Movimiento punto 3  a la izquierda')
    print('W: Movimiento punto 3 a la derecha')
    print('E: Movimiento punto 4 a la izquierda')
    print('R: Movimiento punto 4 a la derecha')
    print('A: Movimiento punto 5  a la izquierda')
    print('D: Movimiento punto 5 a la derecha')
    print('F: Movimiento punto 6 a la izquierda')
    print('G: Movimiento punto 6 a la derecha')
    print('Z: Movimiento punto 7  a la izquierda')
    print('C: Movimiento punto 7 a la derecha')
    print('V: Movimiento punto 8 a la izquierda')
    print('B: Movimiento punto 8 a la derecha')
    print('Y: Movimiento punto 12  a la izquierda')
    print('U: Movimiento punto 12 a la derecha')
    print('H: Movimiento punto 13 a la izquierda')
    print('J: Movimiento punto 13 a la derecha')
    print('N: Movimiento punto 14 a la izquierda')
    print('M: Movimiento punto 14 a la derecha')
    print('I: Movimiento punto 15 a la izquierda')
    print('O: Movimiento punto 15 a la derecha')
    print('P: Previsualizar')
    print('L: Limpiar imagen')
    print('S: Guardar información')
    print('X: Siguiente Imagen')
    print('T: Cerrar Programa')
    print('----------------------------')

archivo="Tesis/coordenadas.csv" #Coordenadas de las imagenes de entrenamiento
#archivo="coordenadas_test.csv" #Coordenadas de las imagenes de prueba
data = pd.DataFrame() 
#bcrear copia de coordenadas.csv, borrar su contenido y dejar vacio para iniciar un nuevo documento
inicio=0 
final=100    #hasta cuantos termina   
######### Indices para la ruta de las imagenes
archivo_indices="Tesis/indices.csv" #Archivo para imagenes de entrenamiento
#archivo_indices="indices_test.csv" #Archivo para imagenes de prueba
data_indices = pd.DataFrame()
data_indices = pd.read_csv(archivo_indices,header=None)
prueba = np.array(data_indices)

for i in range(0,100):# Mover rango para ampliar la seleccion de imagenes
    if prueba[i,1]==1:
       path = "Tesis/COVID-19_Radiography_Dataset/COVID/images/COVID-"+str(prueba[i,2])+'.png'
    if prueba[i,1]==2:
       path = "Tesis/COVID-19_Radiography_Dataset/Normal/images/Normal-"+str(prueba[i,2])+'.png'
    if prueba[i,1]==3:
       path = "Tesis/COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-"+str(prueba[i,2])+'.png'
    
    imagen_original = cv2.imread(path)
    print("Imagen actual:",path) 
    print()
    
    # Crear versión de 64x64 para procesamiento
    imagen_procesamiento = cv2.resize(imagen_original, (64, 64))
    
    # Crear versión más grande para visualización
    imagen_visualizacion = resize_aspect_ratio(imagen_original, width=640)
    
    # Aplicar CLAHE a ambas imágenes
    imagen_procesamiento_clahe = aplicar_clahe(imagen_procesamiento)
    imagen_visualizacion_clahe = aplicar_clahe(imagen_visualizacion)
    
    # Calcular grosor de línea y radio de punto
    grosor_linea = calcular_grosor(imagen_visualizacion_clahe)
    radio_punto = grosor_linea * 2
    
    cv2.namedWindow('Radiografia', cv2.WINDOW_NORMAL)

    
    cv2.setMouseCallback('Radiografia',dibujar)
    cv2.imshow("Radiografia",imagen_visualizacion_clahe)
    cv2.resizeWindow("Radiografia", 640, 640)
    cv2.line(imagen_visualizacion_clahe, (320,0), (320,640), (255,0,0), grosor_linea) #Linea central
    clon = imagen_visualizacion_clahe.copy()
    
    menu()
    
    while True:
        cv2.imshow("Radiografia", imagen_visualizacion_clahe)
        key = cv2.waitKey(1) & 0xFF
        #movimiento de puntos
        if key == ord("q"): #movimiento izquierda punto 3
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord3[0]=coord3[0]-1  #Mover punto
                coord3[1]=int(pendiente*(coord3[0]-cuarto1[0])+cuarto1[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord3[0]=coord3[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        if key == ord("w"): #movimiento derecha punto 3
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord3[0]=coord3[0]+1  #Mover punto
                coord3[1]=int(pendiente*(coord3[0]-cuarto1[0])+cuarto1[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord3[0]=coord3[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("e"): #movimiento izquierda punto 4
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord4[0]=coord4[0]-1  #Mover punto
                coord4[1]=int(pendiente*(coord4[0]-cuarto1[0])+cuarto1[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord4[0]=coord4[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("r"): #movimiento derecha punto 4
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord4[0]=coord4[0]+1  #Mover punto
                coord4[1]=int(pendiente*(coord4[0]-cuarto1[0])+cuarto1[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord4[0]=coord4[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("a"): #movimiento izquierda punto 5
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord5[0]=coord5[0]-1  #Mover punto
                coord5[1]=int(pendiente*(coord5[0]-cuarto2[0])+cuarto2[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord5[0]=coord5[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
            
        if key == ord("d"): #movimiento derecha punto 5
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord5[0]=coord5[0]+1  #Mover punto
                coord5[1]=int(pendiente*(coord5[0]-cuarto2[0])+cuarto2[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord5[0]=coord5[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("f"): #movimiento izquierda punto 6
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord6[0]=coord6[0]-1  #Mover punto
                coord6[1]=int(pendiente*(coord6[0]-cuarto2[0])+cuarto2[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord6[0]=coord6[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
            
        if key == ord("g"): #movimiento derecha punto 6
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord6[0]=coord6[0]+1  #Mover punto
                coord6[1]=int(pendiente*(coord6[0]-cuarto2[0])+cuarto2[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord6[0]=coord6[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("z"): #movimiento izquierda punto 7
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord7[0]=coord7[0]-1  #Mover punto
                coord7[1]=int(pendiente*(coord7[0]-cuarto3[0])+cuarto3[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord7[0]=coord7[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("c"): #movimiento derecha punto 7
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord7[0]=coord7[0]+1  #Mover punto
                coord7[1]=int(pendiente*(coord7[0]-cuarto3[0])+cuarto3[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord7[0]=coord7[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("v"): #movimiento izquierda punto 8
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord8[0]=coord8[0]-1  #Mover punto
                coord8[1]=int(pendiente*(coord8[0]-cuarto3[0])+cuarto3[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord8[0]=coord8[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("b"): #movimiento derecha punto 8
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord8[0]=coord8[0]+1  #Mover punto
                coord8[1]=int(pendiente*(coord8[0]-cuarto3[0])+cuarto3[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord8[0]=coord8[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("y"): #movimiento izquierda punto 12
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord12[0]=coord12[0]-1  #Mover punto
                coord12[1]=int(pendiente*(coord12[0]-inicio[0])+inicio[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord12[0]=coord12[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
    
        if key == ord("u"): #movimiento derecha punto 12
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord12[0]=coord12[0]+1  #Mover punto
                coord12[1]=int(pendiente*(coord12[0]-inicio[0])+inicio[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord12[0]=coord12[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("h"): #movimiento izquierda punto 13
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord13[0]=coord13[0]-1  #Mover punto
                coord13[1]=int(pendiente*(coord13[0]-inicio[0])+inicio[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord13[0]=coord13[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("j"): #movimiento derecha punto 13
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord13[0]=coord13[0]+1  #Mover punto
                coord13[1]=int(pendiente*(coord13[0]-inicio[0])+inicio[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord13[0]=coord13[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("n"): #movimiento izquierda punto 14
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord14[0]=coord14[0]-1  #Mover punto
                coord14[1]=int(pendiente*(coord14[0]-final[0])+final[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord14[0]=coord14[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("m"): #movimiento derecha punto 14
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord14[0]=coord14[0]+1  #Mover punto
                coord14[1]=int(pendiente*(coord14[0]-final[0])+final[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord14[0]=coord14[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)
        
        if key == ord("i"): #movimiento izquierda punto 15
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord15[0]=coord15[0]-1  #Mover punto
                coord15[1]=int(pendiente*(coord15[0]-final[0])+final[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord15[0]=coord15[0]-1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        if key == ord("o"): #movimiento derecha punto 15
            imagen_visualizacion_clahe = clon.copy()
            
            if coord1[0]!=coord2[0]:
                coord15[0]=coord15[0]+1  #Mover punto
                coord15[1]=int(pendiente*(coord15[0]-final[0])+final[1]) #Recalcular
            if coord1[0]==coord2[0]:
                coord15[0]=coord15[0]+1  #Mover punto
            cv2.line(imagen_visualizacion_clahe, tuple(coord1), tuple(coord2), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord3), tuple(coord4), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord5), tuple(coord6), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord7), tuple(coord8), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord12), tuple(coord13), (0,0,255), grosor_linea)
            cv2.line(imagen_visualizacion_clahe, tuple(coord14), tuple(coord15), (0,0,255), grosor_linea)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord1), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord2), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord3), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord4), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord5), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord6), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord7), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord8), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord9), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord10), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord11), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord12), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord13), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord14), radio_punto, (0,255,0), -1)
            cv2.circle(imagen_visualizacion_clahe, tuple(coord15), radio_punto, (0,255,0), -1)

        #Previsualizacion
        if key == ord("p"):
            if todas_coordenadas_definidas():
                # Crear una copia de la imagen de visualización CLAHE
                imagen_preview = imagen_visualizacion_clahe.copy()
                
                # Obtener las dimensiones de la ventana principal
                height, width = imagen_preview.shape[:2]
                
                # Ajustar las coordenadas al tamaño de la ventana de previsualización
                coordenadas = [coord1, coord2, coord3, coord4, coord5, coord6, 
                            coord7, coord8, coord9, coord10, coord11, coord12, coord13, coord14, coord15]
                           
                # Dibujar puntos
                for i, coord in enumerate(coordenadas):
                    cv2.circle(imagen_preview, tuple(coord), radio_punto, (0, 255, 0), -1)
                    cv2.putText(imagen_preview, str(i+1), (coord[0]+5, coord[1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)                
                
                # Mostrar la imagen en una nueva ventana con el mismo tamaño que la ventana principal
                cv2.namedWindow("Previsualización", cv2.WINDOW_NORMAL)
                cv2.imshow("Previsualización", imagen_preview)
                cv2.resizeWindow("Previsualización", width, height)
                
                # Esperar a que se presione una tecla antes de cerrar la ventana de previsualización
                cv2.waitKey(0)
                cv2.destroyWindow("Previsualización")
            else:
                print("Por favor, define todas las coordenadas antes de previsualizar.")
        
        if key == ord("l"): #limpiar la imagen 
            imagen_visualizacion_clahe = clon.copy()
        
        if key == ord("x"):
            break

        if key == ord("s"):
            print('--------------')
            print('Registro Guardado')
            
            # Escalar las coordenadas a la resolución de 64x64
            scale_factor = 64 / imagen_visualizacion_clahe.shape[0]
            coord1_scaled = (int(coord1[0] * scale_factor), int(coord1[1] * scale_factor))
            coord2_scaled = (int(coord2[0] * scale_factor), int(coord2[1] * scale_factor))
            coord3_scaled = (int(coord3[0] * scale_factor), int(coord3[1] * scale_factor))
            coord4_scaled = (int(coord4[0] * scale_factor), int(coord4[1] * scale_factor))
            coord5_scaled = (int(coord5[0] * scale_factor), int(coord5[1] * scale_factor))
            coord6_scaled = (int(coord6[0] * scale_factor), int(coord6[1] * scale_factor))
            coord7_scaled = (int(coord7[0] * scale_factor), int(coord7[1] * scale_factor))
            coord8_scaled = (int(coord8[0] * scale_factor), int(coord8[1] * scale_factor))
            coord9_scaled = (int(coord9[0] * scale_factor), int(coord9[1] * scale_factor))
            coord10_scaled = (int(coord10[0] * scale_factor), int(coord10[1] * scale_factor))
            coord11_scaled = (int(coord11[0] * scale_factor), int(coord11[1] * scale_factor))
            coord12_scaled = (int(coord12[0] * scale_factor), int(coord12[1] * scale_factor))
            coord13_scaled = (int(coord13[0] * scale_factor), int(coord13[1] * scale_factor))
            coord14_scaled = (int(coord14[0] * scale_factor), int(coord14[1] * scale_factor))
            coord15_scaled = (int(coord15[0] * scale_factor), int(coord15[1] * scale_factor))
            
            registrar(archivo, str(i),
                      str(coord1_scaled[0]), str(coord1_scaled[1]),
                      str(coord2_scaled[0]), str(coord2_scaled[1]),
                      str(coord3_scaled[0]), str(coord3_scaled[1]),
                      str(coord4_scaled[0]), str(coord4_scaled[1]),
                      str(coord5_scaled[0]), str(coord5_scaled[1]),
                      str(coord6_scaled[0]), str(coord6_scaled[1]),
                      str(coord7_scaled[0]), str(coord7_scaled[1]),
                      str(coord8_scaled[0]), str(coord8_scaled[1]),
                      str(coord9_scaled[0]), str(coord9_scaled[1]),
                      str(coord10_scaled[0]), str(coord10_scaled[1]),
                      str(coord11_scaled[0]), str(coord11_scaled[1]),
                      str(coord12_scaled[0]), str(coord12_scaled[1]),
                      str(coord13_scaled[0]), str(coord13_scaled[1]),
                      str(coord14_scaled[0]), str(coord14_scaled[1]),
                      str(coord15_scaled[0]), str(coord15_scaled[1]),
                      str(prueba[i,2]))
            break
        if key == ord("t"):
            cv2.destroyAllWindows()
            sys.exit("Programa Terminado")
    cv2.destroyAllWindows()