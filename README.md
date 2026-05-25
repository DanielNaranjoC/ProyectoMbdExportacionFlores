# Modelado temporal de las exportaciones florícolas del Ecuador

Este repositorio contiene el código fuente utilizado para el análisis, visualización, modelado y evaluación predictiva de las exportaciones mensuales de flores naturales del Ecuador, expresadas en miles de dólares FOB.

El proyecto forma parte del trabajo de titulación de la Maestría en Inteligencia de Negocios y Ciencia de Datos, orientado a comparar modelos de pronóstico de series temporales para identificar el enfoque con mejor desempeño predictivo.

## Objetivo del repositorio

Garantizar la trazabilidad y reproducibilidad del análisis desarrollado en el proyecto:

**Modelado temporal de las exportaciones florícolas del Ecuador: evaluación predictiva utilizando enfoques econométricos, de suavizamiento y algorítmicos.**

## Contenido del repositorio

```text
proyecto-exportaciones-flores/
│
├── README.md
├── requirements.txt
├── run_code_models_project.py
├── Base de Datos Flores Naturales.xlsx
```

## Archivo principal

El archivo principal del proyecto es:

```text
run_code_models_project.py
```

Este script ejecuta el flujo completo del análisis:

1. Carga y validación de la base de datos.
2. Revisión de calidad de datos.
3. Generación de gráficos exploratorios.
4. Construcción de la serie mensual.
5. Pruebas de estacionariedad.
6. Prueba de estacionalidad mensual.
7. Estimación de modelos ETS.
8. Estimación de modelos Prophet.
9. Estimación de modelos SARIMAX.
10. Diagnóstico de heterocedasticidad condicional.
11. Estimación de modelos SARIMAX + GARCH(1,1).
12. Comparación de modelos mediante MAE, RMSE y MAPE.
13. Exportación de gráficos y tablas de resultados.

## Fuente de datos

La base utilizada corresponde a una serie mensual de exportaciones de flores naturales del Ecuador, expresada en miles de dólares FOB.

El archivo de entrada esperado por el script es:

```text
Base de Datos Flores Naturales.xlsx
```

El archivo debe contener, como mínimo, las siguientes columnas:

```text
Fecha
Flores naturales
```

## Requisitos

Para instalar las dependencias necesarias, ejecutar:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` debe incluir las librerías necesarias para cargar datos, generar gráficos, entrenar modelos y exportar resultados.

Una versión sugerida es:

```txt
arch
matplotlib
numpy
openpyxl
pandas
prophet
scikit-learn
scipy
seaborn
statsmodels
```

## Ejecución

Para ejecutar el análisis completo:

```bash
python run_code_models_project.py
```

Al finalizar la ejecución, se creará la carpeta:

```text
Resultados Flores/
```

En esta carpeta se guardarán los gráficos en formato SVG y los archivos Excel con las tablas de resultados.

## Modelos evaluados

El análisis compara los siguientes enfoques de modelado:

- Modelos ETS.
- Modelos Prophet aditivo y multiplicativo.
- Modelos SARIMAX.
- Modelos SARIMAX + GARCH(1,1).

## Pruebas estadísticas aplicadas

El script incluye pruebas y diagnósticos para respaldar la selección y evaluación de los modelos:

- ADF.
- Phillips-Perron.
- KPSS.
- DF-GLS.
- Kruskal-Wallis.
- Ljung-Box.
- ARCH-LM.

## Métricas de evaluación

Los modelos son comparados mediante:

- **MAE:** error absoluto medio.
- **RMSE:** raíz del error cuadrático medio.
- **MAPE:** error porcentual absoluto medio.

## Resultados generados

El script genera tres archivos principales de resultados:

```text
pruebas_estadisticas.xlsx
resultados_modelos.xlsx
comparacion_predicciones.xlsx
```

También genera gráficos exploratorios y gráficos de componentes de modelos en formato SVG.

## Reproducibilidad

Para reproducir los resultados:

1. Clonar o descargar este repositorio.
2. Colocar el archivo `Base de Datos Flores Naturales.xlsx` en la raíz del proyecto.
3. Instalar las dependencias con:

   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar:

   ```bash
   python run_code_models_project.py
   ```

5. Revisar la carpeta `Resultados Flores`.

## Uso académico

Este repositorio se incluye como respaldo técnico del proyecto de titulación. Su finalidad es documentar el procedimiento computacional utilizado para la limpieza, visualización, modelado y evaluación predictiva de la serie temporal analizada.

## Cita sugerida

Naranjo Carranza, J. D., & Redrobán Corrales, E. L. (2026). *Código fuente del proyecto: Modelado temporal de las exportaciones florícolas del Ecuador* [Repositorio de GitHub]. GitHub. https://github.com/usuario/nombre-del-repositorio

## Autores

Jonathan Daniel Naranjo Carranza  
Emily Lissette Redrobán Corrales

## Licencia

Este repositorio se publica con fines académicos y de reproducibilidad del proyecto de titulación.
