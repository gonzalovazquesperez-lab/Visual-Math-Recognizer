# Visual Math Recognizer 

Calculadora interactiva que utiliza visión artificial para traducir lenguaje de señas en operaciones matemáticas en tiempo real. 

*Descripción General
Este sistema procesa la entrada de video mediante la cámara web, detecta los gestos de las manos y los traduce en comandos matemáticos. Incorpora también retroalimentación auditiva para una experiencia más accesible y fluida. 

Debido a los requerimientos de procesamiento visual en tiempo real, el proyecto se distribuye en su versión de código fuente nativo para garantizar la máxima compatibilidad de hardware.

*Cómo ejecutar el proyecto

No es necesario realizar configuraciones complejas ni instalaciones manuales. Sigue estos pasos:

1. Descarga este repositorio haciendo clic en el botón verde **"Code"** y luego en **"Download ZIP"**.
2. Descomprime la carpeta en tu computadora.
3. Haz doble clic en el archivo **`Ejecutar.bat`**.

El sistema se encargará automáticamente de leer el archivo `requirements.txt`, instalar las librerías necesarias de visión artificial y ejecutar la calculadora.

*Tecnologías y Librerías Utilizadas
- **Python 3** - Lenguaje base.
- **OpenCV (`opencv-python`)** - Captura y procesamiento de video en tiempo real.
- **MediaPipe** - Detección y seguimiento de los puntos de articulación de las manos.
- **NumPy** - Operaciones y cálculos de matrices.
- **Pyttsx3** - Síntesis de voz para la lectura de resultados.
