# AI4ES2023_CHALLENGE_A
Código del Datathon de AI4ES 2023 para el reto A
## Requisitos
Para la ejecución de este código es necesario tener instalado Python 3.7 o superior, además de las librerías que se encuentran en el archivo requirements.txt. Para instalarlas, ejecutar el siguiente comando en la terminal:
```
pip install -r requirements.txt
```
## Estructura del código
El código se divide en 3 ficheros:
* `main.py`: Fichero principal, que contiene el código para la ejecución del reto.
* `utils.py`: Fichero con funciones auxiliares para la ejecución del reto.
* `train.py`: Fichero con funciones auxiliares para el entrenamiento de los modelos.
* `test.py`: Fichero con funciones auxiliares para la evaluación de los modelos.
* `api`: Carpeta con el código para la estracción y reconstrucción de los datos. Además de contener las metricas de evaluación.

## Ejecución
Para ejecutar el código, tendremos las siguientes opciones, dentro del fichero main.py:
* `--mode`: Modo de ejecución del código. Puede ser `train` o `test`. Por defecto, es `train`.
* `--model_EMU`: Extractor de características para el modelo EMU. Puede ser `"MobileNetV2","ResNet50","comb" `. Por defecto, es `comb`.
* `--model_ANOMALY`: Algoritmo de detección de anomalías para el modelo ANOMALY. Puede ser `"IF", "AE","comb" `. Por defecto, es `comb`.
* `--window_size`: Tamaño de la ventana de datos. Por defecto, es `16`.
* `--step_size`: Tamaño del paso de la ventana de datos. Por defecto, es `8`.
* `--batch_size`: Tamaño del batch de datos. Por defecto, es `2048`.