# Simulación del caudal en España utilizando redes Long Short-Term Memory

## Introducción

Trabajo final del Máster en Ciencia de Datos de la Universidad de Alcalá de Henares en el curso 2022-2023.

El trabajo tiene 3 objetivos:
1. Crear el conjunto de datos CAMELS-ES (_Catchment Attributes and Meteorology for Large-sample Studies - España_) . Este conjunto se enmarca dentro de la iniciativa CARAVAN para crear muestras a gran escala de cuencas hidrológicas, incluyendo series temporales diarias (meteorología y caudal) y atributos estáticos (geomorfología, usos del suelo, tipo de suelo, vegetación...).
2. Crear una red neuronal recurrente de tipo LSTM (_Long-short Term Memory_) basada en los datos de CAMELS-ES y capaz de simular el caudal diario en cualquier punto de la España Peninsular.
3. Crear una segunda red LSTM capaz de emular al modelo hidrológico LISFLOOD-OS, el utilizado en el sistema EFAS (_European Flood Awareness System_). Para ello ha de expandirse primero el conjunto de datos CAMELS-ES con los datos de entrada del modelo LISFLOOD-OS (series meteorológicas, mapas estáticos y parámetros calibrados del modelo). Posteriormente se entrena una red LSTM capaz de replicar el caudal simulado por LISFLOOD-OS con sus mismos datos de entrada.

## Organización

Actualmente el repositorio cuenta con cinco directorios:

1. `bib` contiene algunas referencias bibliográficas.
2. `data` contiene los datos de partida utilizados para generar el conjunto de datos CAMELS-ES, así como el resultado final.
3. `docs` contiene documentos como la propuesta de trabajo y el informe final.
4. `environment` contiene los entornos Conda necesarios para replicar los códigos utilizado.
5. `notebooks` contiene los cuadernos de Jupyter utilizados en las diversas fases del estudio.
