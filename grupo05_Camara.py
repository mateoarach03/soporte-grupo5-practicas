import cv2
import time

def main():
    # Inicialización de la cámara web
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return

    # Obtener resolución de la cámara
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    
    grabando = False
    tiempo_limite_foto = 0
    segundos_timer = 3.5  

    print("Iniciando control de cámara. Presiona 'Q' en la ventana de video para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo recibir el fotograma de la cámara.")
            break

        frame_mostrar = frame.copy()

        # 1. Temporizador para las fotos
        if tiempo_limite_foto > 0:
            tiempo_restante = tiempo_limite_foto - time.time()
            if tiempo_restante > 0:
                # Mostrar cuenta regresiva 
                texto_timer = f"Foto en: {int(tiempo_restante) + 1}s"
                cv2.putText(frame_mostrar, texto_timer, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3, cv2.LINE_AA)
            else:
                # Tomar foto
                nombre_foto = f"foto_timer_{int(time.time())}.png"
                cv2.imwrite(nombre_foto, frame)
                print(f"Foto capturada con temporizador: {nombre_foto}")
                tiempo_limite_foto = 0  # Resetear temporizador

        # 2. Grabación de video
        if grabando:
            if out is not None:
                out.write(frame)
            
            if int(time.time()) % 2 == 0:
                cv2.circle(frame_mostrar, (30, 35), 10, (0, 0, 255), -1)
            else:
                cv2.circle(frame_mostrar, (30, 35), 10, (0, 0, 100), -1)  # Rojo oscuro
                
            cv2.putText(frame_mostrar, "REC", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 3. Menú de controles
        texto_controles = "V: Grabar | F: Foto | T: Timer | Q: Salir"
        cv2.putText(frame_mostrar, texto_controles, (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_mostrar, texto_controles, (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Control de Camara - Ejercicio", frame_mostrar)

        # 4. Captura del teclado (espera 1 ms)
        tecla = cv2.waitKey(1) & 0xFF

        # Q: Salir
        if tecla == ord('q') or tecla == ord('Q'):
            break

        # F: Foto instantánea
        elif tecla == ord('f') or tecla == ord('F'):
            nombre_foto = f"foto_instantanea_{int(time.time())}.png"
            cv2.imwrite(nombre_foto, frame)
            print(f"Foto instantánea guardada: {nombre_foto}")

        # T: Activar temporizador
        elif tecla == ord('t') or tecla == ord('T'):
            tiempo_limite_foto = time.time() + segundos_timer
            print(f"Temporizador iniciado ({segundos_timer} segundos)...")

        # V: Iniciar/Detener grabación
        elif tecla == ord('v') or tecla == ord('V'):
            grabando = not grabando
            if grabando:
                nombre_video = f"video_{int(time.time())}.avi"
                out = cv2.VideoWriter(nombre_video, fourcc, 20.0, (width, height))
                print(f"Grabando video: {nombre_video}")
            else:
                if out is not None:
                    out.release()
                    out = None
                print("Grabación finalizada y guardada.")

    # Limpieza de recursos
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")

if __name__ == "__main__":
    main()
