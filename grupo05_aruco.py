import cv2
import numpy as np


# Matriz intrínseca de la cámara (aproximada para resoluciones estándar de webcams)
camera_matrix = np.array([[800.0, 0.0, 320.0],
                          [0.0, 800.0, 240.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

# Coeficientes de distorsión de la lente (se asumen nulos para simplificar)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Parámetros físicos del marcador ArUco (lado de 5 cm)
marker_length = 0.05 
half_l = marker_length / 2.0

# Puntos 3D que definen la pirámide (4 en la base en Z=0, y el vértice superior en Z=-0.05)
pyramid_points = np.array([
    [-half_l,  half_l, 0.0],               
    [ half_l,  half_l, 0.0],               
    [ half_l, -half_l, 0.0],               
    [-half_l, -half_l, 0.0],               
    [ 0.0,     0.0,    -marker_length]     
], dtype=np.float32)

# Posición 3D de las esquinas del marcador para la estimación de la pose
marker_3d_edges = np.array([
    [-half_l,  half_l, 0.0],
    [ half_l,  half_l, 0.0],
    [ half_l, -half_l, 0.0],
    [-half_l, -half_l, 0.0]
], dtype=np.float32)



 # diccionario predefinido DICT_6X6_250
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Inicializar captura de video
cap = cv2.VideoCapture(0)
window_name = 'Realidad Aumentada - Grupo 05'
cv2.namedWindow(window_name)

print("Iniciando aplicación de Realidad Aumentada...")
print("Presiona 'q', 'Q' o 'ESC' para salir de la aplicación.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame de la cámara.")
        break

    # Conversión a escala de grises para la detección de marcadores
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # Si se detectaron marcadores
    if ids is not None:
        for i in range(len(ids)):
            # Estimar la pose del marcador (rotación y traslación en 3D)
            success, rvec, tvec = cv2.solvePnP(
                marker_3d_edges, corners[i], camera_matrix, dist_coeffs
            )

            if success:
                # Proyectar los puntos 3D de la pirámide al plano 2D de la imagen
                img_points, _ = cv2.projectPoints(
                    pyramid_points, rvec, tvec, camera_matrix, dist_coeffs
                )
                
                # Convertir los puntos proyectados a coordenadas enteras
                pts = np.int32(img_points).reshape(-1, 2)

                # Clonar el frame original para mezclar la transparencia
                overlay = frame.copy()

                # Dibujar las caras de la pirámide con colores semi-transparentes
                # Cara 1: Roja (Frente/Superior)
                cv2.fillConvexPoly(overlay, np.array([pts[0], pts[1], pts[4]]), (0, 0, 255))
                # Cara 2: Verde (Derecha)
                cv2.fillConvexPoly(overlay, np.array([pts[1], pts[2], pts[4]]), (0, 255, 0))
                # Cara 3: Azul (Atrás/Inferior)
                cv2.fillConvexPoly(overlay, np.array([pts[2], pts[3], pts[4]]), (255, 0, 0))
                # Cara 4: Amarilla (Izquierda)
                cv2.fillConvexPoly(overlay, np.array([pts[3], pts[0], pts[4]]), (0, 255, 255))

                # Mezclar la capa de caras transparentes (50% de opacidad) con el frame
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # Dibujar las aristas (bordes) de la base de la pirámide en color blanco
                cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 2)
                cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (255, 255, 255), 2)
                cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (255, 255, 255), 2)
                cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (255, 255, 255), 2)
                
                # Dibujar las aristas que conectan la base con el vértice superior
                for j in range(4):
                    cv2.line(frame, tuple(pts[j]), tuple(pts[4]), (255, 255, 255), 2)

                # Dibujar los ejes coordenados (X: rojo, Y: verde, Z: azul) en el origen del marcador
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

    # Mostrar la imagen resultante
    cv2.imshow(window_name, frame)

    # Procesar teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        break

    # Verificar si el usuario cerró la ventana usando la interfaz gráfica
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

for _ in range(10):
    cv2.waitKey(1)
