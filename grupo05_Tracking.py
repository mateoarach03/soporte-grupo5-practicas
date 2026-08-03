import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        """
        Inicializa el rastreador de centroides.
        :param max_disappeared: Cantidad de fotogramas que un objeto puede estar ausente
                                antes de ser eliminado del rastreo.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()      # id -> centroide (x, y)
        self.boxes = OrderedDict()        # id -> bounding box (startX, startY, endX, endY)
        self.disappeared = OrderedDict()  # id -> contador de fotogramas ausente
        self.paths = OrderedDict()        # id -> lista de centroides históricos (últimos 30)
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        """Registra un nuevo objeto con su centroide y caja delimitadora."""
        self.objects[self.next_object_id] = centroid
        self.boxes[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        self.paths[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        """Elimina el objeto del rastreo."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.boxes:
            del self.boxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        if object_id in self.paths:
            del self.paths[object_id]

    def update(self, rects):
        """
        Actualiza las posiciones de los objetos rastreados en base a nuevas detecciones.
        :param rects: Lista de cajas delimitadoras [(startX, startY, endX, endY), ...]
        :return: Diccionario de objetos activos con sus centroides.
        """
        # Si no hay detecciones en este fotograma, incrementar la ausencia de todos los objetos
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calcular los centroides de las nuevas detecciones
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # Si no estamos rastreando ningún objeto, registrar todas las detecciones
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            # Obtener IDs y centroides actualmente rastreados
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Calcular matriz de distancias euclidianas entre centroides existentes y nuevos
            # Usamos broadcasting de numpy para evitar dependencias externas como scipy
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            # Para asociar, buscamos la menor distancia por fila y ordenamos
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Vincular detecciones existentes con las nuevas según menor distancia
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.boxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                
                # Agregar al camino y limitar historial
                self.paths[object_id].append(input_centroids[col])
                if len(self.paths[object_id]) > 30:
                    self.paths[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            # Identificar filas y columnas no utilizadas
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Si hay más objetos rastreados que detecciones, marcar los no detectados como ausentes
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            # Si hay más detecciones que objetos rastreados, registrar nuevas detecciones
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects[col])

        return self.objects


# --- FUNCIONES AUXILIARES DE MATEMÁTICA VECTORIAL ---

def ccw(A, B, C):
    """Verifica si los puntos A, B y C están en sentido antihorario (Counter-Clockwise)."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersection(p1, p2, a, b):
    """
    Determina si el segmento p1-p2 (movimiento) se cruza con el segmento a-b (línea de conteo).
    """
    return ccw(p1, a, b) != ccw(p2, a, b) and ccw(p1, p2, a) != ccw(p1, p2, b)

def get_crossing_direction(p1, p2, a, b):
    """
    Calcula la dirección de cruce usando el producto punto del vector de movimiento
    y el vector normal a la línea de conteo.
    Retorna +1 para un sentido (IN) y -1 para el opuesto (OUT).
    """
    # Vector de movimiento del objeto: v = p2 - p1
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    
    # Vector de la línea de conteo: u = b - a
    ux = b[0] - a[0]
    uy = b[1] - a[1]
    
    # Vector normal a la línea de conteo: n = (-uy, ux)
    nx = -uy
    ny = ux
    
    # Producto punto
    dot = vx * nx + vy * ny
    return 1 if dot > 0 else -1


# --- CONFIGURACIÓN DE LÍNEA MEDIANTE MOUSE ---

line_pts = []
drawing = False
temp_pt = None

def draw_line_callback(event, x, y, flags, param):
    global line_pts, drawing, temp_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        line_pts = [(x, y)]
        drawing = True
        temp_pt = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if len(line_pts) == 1:
            line_pts.append((x, y))
            drawing = False
            temp_pt = None

def setup_counting_line(cap, target_width=1000):
    """
    Muestra el primer fotograma del video y permite dibujar la línea de conteo interactiva.
    """
    global line_pts, drawing, temp_pt
    
    # Resetear captura al inicio
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_first = cap.read()
    if not ret:
        # Reintentar un par de veces si es webcam
        for _ in range(5):
            ret, frame_first = cap.read()
        if not ret:
            print("Error: No se pudo capturar el fotograma inicial.")
            return None
            
    # Redimensionar el fotograma inicial para que coincida con el espacio de trabajo del bucle principal
    h, w = frame_first.shape[:2]
    if w > target_width:
        scale = target_width / w
        new_h = int(h * scale)
        frame_first = cv2.resize(frame_first, (target_width, new_h))
        h, w = new_h, target_width

    window_name = "Definir Linea de Conteo"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_line_callback)
    
    print("\n[CONFIGURACIÓN DE LÍNEA DE CONTEO]")
    print(" -> Mantén click izquierdo presionado para iniciar la línea, arrastra y suelta para terminar.")
    print(" -> Presiona ESPACIO o ENTER para confirmar la línea dibujada.")
    print(" -> Presiona 'D' para usar una línea horizontal por defecto (al 60% de altura).")
    print(" -> Presiona ESC o 'Q' para salir.")
    
    line_pts = []
    
    while True:
        temp_frame = frame_first.copy()
        
        # Dibujar el estado actual del dibujo de la línea
        if len(line_pts) == 2:
            cv2.line(temp_frame, line_pts[0], line_pts[1], (0, 255, 0), 3, cv2.LINE_AA)
            cv2.circle(temp_frame, line_pts[0], 6, (0, 0, 255), -1)
            cv2.circle(temp_frame, line_pts[1], 6, (0, 0, 255), -1)
        elif len(line_pts) == 1 and temp_pt is not None:
            cv2.line(temp_frame, line_pts[0], temp_pt, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(temp_frame, line_pts[0], 6, (0, 0, 255), -1)
            
        # Dibujar textos informativos en pantalla
        cv2.putText(temp_frame, "Arrastra con click IZQ para dibujar la linea", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(temp_frame, "ESPACIO / ENTER: Confirmar | D: Por defecto | Q: Salir", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
                    
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(15) & 0xFF
        
        if key == 32 or key == 13:  # ESPACIO o ENTER
            if len(line_pts) == 2:
                break
            else:
                print("Dibuja una línea antes de continuar o presiona 'D'.")
        elif key == ord('d') or key == ord('D'):
            # Línea horizontal por defecto en la parte media-baja
            line_pts = [(50, int(h * 0.65)), (w - 50, int(h * 0.65))]
            break
        elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC o Q
            line_pts = []
            break
            
    cv2.destroyWindow(window_name)
    return line_pts if len(line_pts) == 2 else None


# --- SELECCIÓN DE ARCHIVO DE VIDEO ---

def select_video_path():
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Mostrar el diálogo al frente
        file_path = filedialog.askopenfilename(
            title="Selecciona el video para el Tracking",
            filetypes=[
                ("Archivos de Video", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Todos los archivos", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"Error al abrir el selector de archivos: {e}")
        return None


# --- FUNCIÓN PRINCIPAL ---

def main():
    print("Iniciando aplicación de Tracking y Conteo de Objetos...")
    
    # 1. Seleccionar video
    video_path = select_video_path()
    if not video_path:
        print("No se seleccionó ningún archivo de video en el cuadro de diálogo.")
        print("Ingresa la ruta de tu video o presiona Enter para usar la Cámara Web:")
        input_path = input("Ruta del video: ").strip()
        if input_path == "":
            video_path = 0
            print("Configurando cámara web...")
        else:
            video_path = input_path
            if not os.path.exists(video_path):
                print(f"Error: El archivo '{video_path}' no existe. Saliendo.")
                return

    # 2. Inicializar VideoCapture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video: {video_path}")
        return

    # 3. Configurar línea de conteo interactiva
    target_width = 1000
    line_pts_selected = setup_counting_line(cap, target_width)
    if not line_pts_selected:
        print("Configuración cancelada o sin línea de conteo. Finalizando.")
        cap.release()
        return
        
    line_pt1, line_pt2 = line_pts_selected
    print(f"Línea establecida correctamente: {line_pt1} -> {line_pt2}")

    # 4. Inicializar variables de procesamiento
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    tracker = CentroidTracker(max_disappeared=25)
    
    # Contadores
    count_in = 0
    count_out = 0
    counted_objects = {}  # id -> último sentido de cruce ("IN" o "OUT")
    
    # Control de destello de línea (feedback visual al cruzar)
    flash_timer = 0 
    
    # Parámetros de detección fijos
    min_area = 800
    max_area = 60000
    bin_threshold = 45

    # Elemento estructurante para operaciones morfológicas
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    pausado = False
    
    print("\n[CONTROLES DE REPRODUCCIÓN]")
    print(" -> 'Q' o ESC: Salir del programa")
    print(" -> 'P': Pausar / Reanudar video")
    print(" -> 'R': Reiniciar contadores a cero")
    print(" -> 'L': Redibujar línea de conteo")

    # Reiniciar video a fotograma cero por si acaso
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                # Si el video terminó, pausar para permitir ver el resultado final
                if video_path != 0:
                    print("Fin del video alcanzado.")
                    pausado = True
                    continue
                else:
                    print("Error al recibir fotograma de la cámara.")
                    break

            # Redimensionar frame para consistencia
            h, w = frame.shape[:2]
            if w > target_width:
                scale = target_width / w
                new_h = int(h * scale)
                frame = cv2.resize(frame, (target_width, new_h))
                h, w = new_h, target_width

            # --- PROCESAMIENTO DE IMAGEN PARA DETECCIÓN ---
            # Aplicar sustracción de fondo
            fg_mask = bg_subtractor.apply(frame)

            # Umbralizar para remover sombras (píxeles grises 127 de MOG2)
            _, fg_mask = cv2.threshold(fg_mask, bin_threshold, 255, cv2.THRESH_BINARY)

            # Limpieza morfológica de ruido
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

            # Buscar contornos
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Obtener caja delimitadora
                    (x, y, box_w, box_h) = cv2.boundingRect(contour)
                    rects.append((x, y, x + box_w, y + box_h))

            # --- ACTUALIZAR RASTREADOR DE OBJETOS ---
            tracked_objects = tracker.update(rects)

            # --- PROCESAR CADA OBJETO RASTREADO ---
            for (object_id, centroid) in tracked_objects.items():
                # Obtener bounding box del objeto
                box = tracker.boxes[object_id]
                startX, startY, endX, endY = box

                # Dibujar bounding box del objeto rastreado (Azul)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 120, 0), 2)
                
                # Dibujar centroide actual
                cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 255), -1)
                
                # Dibujar etiqueta de ID del objeto
                cv2.putText(frame, f"ID: {object_id}", (startX, startY - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Dibujar la trayectoria histórica (línea de puntos amarilla)
                path = tracker.paths[object_id]
                for i in range(1, len(path)):
                    cv2.line(frame, path[i - 1], path[i], (0, 255, 255), 1, cv2.LINE_AA)

                # --- VERIFICAR CRUCE DE LÍNEA ---
                if len(path) >= 2:
                    p1 = path[-2]  # Centroide en fotograma anterior
                    p2 = path[-1]  # Centroide en fotograma actual

                    # Si el segmento p1-p2 cruza la línea de conteo
                    if check_intersection(p1, p2, line_pt1, line_pt2):
                        # Calcular dirección (IN/OUT)
                        direction = get_crossing_direction(p1, p2, line_pt1, line_pt2)
                        side = "IN" if direction > 0 else "OUT"

                        # Evitar contar el mismo lado de forma consecutiva
                        if object_id not in counted_objects or counted_objects[object_id] != side:
                            if side == "IN":
                                count_in += 1
                                print(f"[INFO] Objeto ID {object_id} cruzó hacia ADENTRO (IN).")
                            else:
                                count_out += 1
                                print(f"[INFO] Objeto ID {object_id} cruzó hacia AFUERA (OUT).")
                            
                            counted_objects[object_id] = side
                            flash_timer = 8  # Destello de 8 fotogramas

            # --- DIBUJAR LÍNEA DE CONTEO ---
            # Si hubo cruce reciente, destella la línea en rojo, sino se muestra en verde
            if flash_timer > 0:
                line_color = (0, 0, 255)  # Rojo
                line_thickness = 4
                flash_timer -= 1
            else:
                line_color = (0, 255, 0)  # Verde
                line_thickness = 2
                
            cv2.line(frame, line_pt1, line_pt2, line_color, line_thickness, cv2.LINE_AA)
            cv2.circle(frame, line_pt1, 5, (0, 0, 255), -1)
            cv2.circle(frame, line_pt2, 5, (0, 0, 255), -1)

            # Escribir indicador del sentido "IN/OUT" en los extremos de la línea
            cv2.putText(frame, "IN (+)", (line_pt1[0] + 10, line_pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "OUT (-)", (line_pt2[0] - 50, line_pt2[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # --- DIBUJAR HUD/DASHBOARD ---
            # Crear overlay semitransparente para panel de información (esquina superior izquierda)
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (320, 160), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Textos del Dashboard
            cv2.putText(frame, "DASHBOARD DE SEGUIMIENTO", (25, 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Entradas (IN): {count_in}", (30, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Salidas (OUT): {count_out}", (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Objetos Activos: {len(tracked_objects)}", (30, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 200, 0), 1, cv2.LINE_AA)

            # Barra inferior con controles rápidos
            cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
            control_text = "Q: Salir | P: Pausar | R: Reiniciar Conteo | L: Redibujar Linea"
            cv2.putText(frame, control_text, (20, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # --- MOSTRAR VENTANA ---
            cv2.imshow("Tracking y Conteo - OpenCV", frame)

        # --- CAPTURA DE TECLADO ---
        key = cv2.waitKey(20) & 0xFF

        # Salir (Q o ESC)
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Cerrando la aplicación...")
            break

        # Pausar (P)
        elif key == ord('p') or key == ord('P'):
            pausado = not pausado
            estado = "PAUSADO" if pausado else "REPRODUCIENDO"
            print(f"[CONTROL] Reproducción: {estado}")

        # Reiniciar conteo (R)
        elif key == ord('r') or key == ord('R'):
            count_in = 0
            count_out = 0
            counted_objects.clear()
            print("[CONTROL] Contadores reiniciados a cero.")

        # Redibujar línea (L)
        elif key == ord('l') or key == ord('L'):
            print("[CONTROL] Abriendo diálogo para redibujar línea de conteo...")
            # Pausar temporalmente
            temp_pause = pausado
            pausado = True
            
            new_line = setup_counting_line(cap, target_width)
            if new_line:
                line_pt1, line_pt2 = new_line
                count_in = 0
                count_out = 0
                counted_objects.clear()
                print(f"[CONTROL] Nueva línea guardada: {line_pt1} -> {line_pt2}. Contadores reiniciados.")
            
            pausado = temp_pause

    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    print("Aplicación finalizada correctamente.")

if __name__ == "__main__":
    main()
