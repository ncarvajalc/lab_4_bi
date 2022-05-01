# Laboratorio 4 - Inteligencia de Negocios

Este laboratorio tiene como objetivo profundizar en la construcción de pipelines con el fin de llevar modelos de machine learning a producción. Además, se busca construir un API para montar el modelo y realizar predicciones mediante peticiones HTTP.

## Instalación API

Para poder ejecutar el API del proyecto tendremos que descargar las dependencias requeridas, para esto utilice el comando:

`pip install -r requirements.txt`

## Despliegue API

Una vez descargadas las dependecias del proyecto, tenemos que ejecutar el servidor utilizando el comando:

`uvicorn main:app --reload`

## Funcionamiento API

Para el funcionamiento del API haremos uso de dos endpoints, los cuales son:

- urlservidor/predict
- urlservidor/predictMany

El endpoint correspondiente a **/predict** se encarga de realizar la predicción para un caso. Por otra parte, el endpoint **/predictMany** se encarga de realizar el R^2 del modelo.
