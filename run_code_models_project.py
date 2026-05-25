# -*- coding: utf-8 -*-
"""
Análisis, gráficos y modelado de exportaciones de flores naturales.

El script unifica el análisis local de calidad de datos, la generación de
gráficos exploratorios y el entrenamiento/comparación de modelos de series
temporales para exportaciones mensuales de flores naturales.

Antes de ejecutarlo, asegúrate de tener instaladas las dependencias externas
necesarias en tu entorno local: pandas, numpy, matplotlib, seaborn, scipy,
statsmodels, scikit-learn, openpyxl, arch y prophet.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
from arch.unitroot import ADF, DFGLS, KPSS, PhillipsPerron
from prophet import Prophet
from scipy.stats import kruskal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


INPUT_FILE = Path("Base de Datos Flores Naturales.xlsx")
OUTPUT_DIR = Path("Resultados Flores")

DATE_COLUMN = "fecha"
VALUE_COLUMN = "flores_naturales"
ORIGINAL_DATE_COLUMN = "Fecha"
ORIGINAL_VALUE_COLUMN = "Flores naturales"

NEW_DATA_VALUES = {
    datetime(2026, 2, 1): 127035.284965,
    datetime(2026, 3, 1): 93454.625564,
}

MONTH_NAMES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

ALL_SARIMA_MODELS = {
    "Model 1": {"AR": [1], "MA": [1], "SAR": [], "SMA": []},
    "Model 2": {"AR": [1, 2], "MA": [1, 2], "SAR": [], "SMA": []},
    "Model 3": {"AR": [1, 2, 7], "MA": [1], "SAR": [], "SMA": []},
    "Model 4": {
        "AR": [1, 2, 10],
        "MA": [1, 2, 10],
        "SAR": [],
        "SMA": [],
    },
    "Model 5": {"AR": [1, 2, 11], "MA": [1, 2], "SAR": [], "SMA": []},
    "Model 6": {"AR": [1], "MA": [1], "SAR": [], "SMA": [1]},
    "Model 7": {"AR": [2], "MA": [2], "SAR": [1], "SMA": [1]},
    "Model 8": {"AR": [1, 2], "MA": [1, 2], "SAR": [1], "SMA": [1]},
    "Model 9": {"AR": [1], "MA": [1, 2], "SAR": [1], "SMA": []},
    "Model 10": {"AR": [2], "MA": [1], "SAR": [1], "SMA": [1]},
    "Model 11": {"AR": [2, 10], "MA": [1, 10], "SAR": [1], "SMA": [1]},
    "Model 12": {"AR": [1], "MA": [1], "SAR": [1], "SMA": [1]},
    "Model 13": {
        "AR": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "MA": [2, 4, 6, 8],
        "SAR": [],
        "SMA": [1],
    },
}

SELECTED_SARIMA_MODELS = {
    "Model 10": ALL_SARIMA_MODELS["Model 10"],
    "Model 13": ALL_SARIMA_MODELS["Model 13"],
}


def configure_environment() -> None:
    """
    Configura opciones generales para la ejecución del script.
    """
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", context="talk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(input_file: Path) -> pd.DataFrame:
    """
    Carga y estandariza la base de exportaciones de flores naturales.

    Parameters
    ----------
    input_file : pathlib.Path
        Ruta del archivo Excel con la base mensual.

    Returns
    -------
    pandas.DataFrame
        Base con columnas estandarizadas: fecha y flores_naturales.

    Raises
    ------
    FileNotFoundError
        Si el archivo de entrada no existe.
    KeyError
        Si no se encuentran las columnas esperadas.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_file}")

    data = pd.read_excel(input_file, sheet_name=0)
    data.columns = data.columns.str.strip()

    missing_columns = {
        ORIGINAL_DATE_COLUMN,
        ORIGINAL_VALUE_COLUMN,
    }.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Faltan columnas requeridas en el Excel: {missing}")

    data = data.rename(
        columns={
            ORIGINAL_DATE_COLUMN: DATE_COLUMN,
            ORIGINAL_VALUE_COLUMN: VALUE_COLUMN,
        }
    )

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors="coerce")

    return data


def print_data_quality(data: pd.DataFrame) -> None:
    """
    Imprime un análisis básico de calidad, rango y tamaño de la base.

    El análisis incluye número de registros, duplicados, valores faltantes,
    fecha inicial, fecha final y meses faltantes dentro de la secuencia
    mensual esperada.

    Parameters
    ----------
    data : pandas.DataFrame
        Base con columnas fecha y flores_naturales.
    """
    valid_dates = data[DATE_COLUMN].dropna()
    missing_values = data.isna().sum()
    duplicated_rows = data.duplicated().sum()

    print("=" * 70)
    print("ANÁLISIS BÁSICO DE LA BASE")
    print("=" * 70)
    print(f"Número de datos: {len(data)}")
    print(f"Registros duplicados: {duplicated_rows}")
    print("-" * 70)
    print("Valores faltantes por columna:")
    print(missing_values.to_string())

    if valid_dates.empty:
        print("-" * 70)
        print("No existen fechas válidas para calcular el rango temporal.")
        print("=" * 70)
        return

    date_min = valid_dates.min()
    date_max = valid_dates.max()
    expected_dates = pd.date_range(start=date_min, end=date_max, freq="MS")
    missing_months = expected_dates.difference(valid_dates)

    print("-" * 70)
    print(f"Fecha inicial: {date_min.date()}")
    print(f"Fecha final: {date_max.date()}")
    print(f"Rango temporal: {date_min.date()} a {date_max.date()}")
    print(f"Meses faltantes en la secuencia mensual: {len(missing_months)}")

    if len(missing_months) > 0:
        print("-" * 70)
        print("Meses faltantes:")
        print(missing_months.strftime("%Y-%m-%d").to_list())

    print("=" * 70)


def clean_data_for_plots(data: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y ordena la base para la generación de gráficos y modelos.

    Parameters
    ----------
    data : pandas.DataFrame
        Base original con posibles valores faltantes.

    Returns
    -------
    pandas.DataFrame
        Base sin valores faltantes en fecha y flores_naturales.
    """
    clean_data = data.dropna(subset=[DATE_COLUMN, VALUE_COLUMN])
    clean_data = clean_data.sort_values(DATE_COLUMN).reset_index(drop=True)

    return clean_data


def add_plot_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables auxiliares para gráficos exploratorios.

    Parameters
    ----------
    data : pandas.DataFrame
        Base con columnas fecha y flores_naturales.

    Returns
    -------
    pandas.DataFrame
        Base con año, mes, nombre de mes y media móvil de 12 meses.
    """
    plot_data = data.copy()
    plot_data["anio"] = plot_data[DATE_COLUMN].dt.year
    plot_data["mes"] = plot_data[DATE_COLUMN].dt.month
    plot_data["mes_nombre"] = plot_data["mes"].map(MONTH_NAMES)
    plot_data["media_movil_12"] = (
        plot_data[VALUE_COLUMN].rolling(window=12, min_periods=12).mean()
    )

    return plot_data


def build_monthly_series(data: pd.DataFrame) -> pd.Series:
    """
    Construye la serie mensual usada en los modelos.

    Parameters
    ----------
    data : pandas.DataFrame
        Base limpia con fecha y flores_naturales.

    Returns
    -------
    pandas.Series
        Serie mensual indexada por fecha de inicio de mes.
    """
    monthly_series = (
        data.set_index(DATE_COLUMN)[VALUE_COLUMN]
        .sort_index()
        .asfreq("MS")
        .dropna()
    )

    return monthly_series


def save_current_plot(output_path: Path) -> None:
    """
    Guarda el gráfico activo y libera memoria.

    Parameters
    ----------
    output_path : pathlib.Path
        Ruta final donde se guardará la imagen.
    """
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """
    Guarda una figura específica y libera memoria.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura que se debe guardar.
    output_path : pathlib.Path
        Ruta final donde se guardará la imagen.
    """
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_moving_average(data: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera el gráfico 02: serie temporal con media móvil de 12 meses.

    Parameters
    ----------
    data : pandas.DataFrame
        Base con fecha, flores_naturales y media_movil_12.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    plt.figure(figsize=(16, 6))
    sns.lineplot(
        data=data,
        x=DATE_COLUMN,
        y=VALUE_COLUMN,
        label="Serie original",
    )
    sns.lineplot(
        data=data,
        x=DATE_COLUMN,
        y="media_movil_12",
        label="Media móvil 12 meses",
    )
    plt.title("Serie temporal y media móvil de 12 meses")
    plt.xlabel("Fecha")
    plt.ylabel("Miles de dólares FOB")

    save_current_plot(output_dir / "02_serie_media_movil.svg")


def plot_monthly_boxplot(data: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera el gráfico 03: boxplot mensual de exportaciones.

    Parameters
    ----------
    data : pandas.DataFrame
        Base con mes_nombre y flores_naturales.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    month_order = list(MONTH_NAMES.values())

    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=data,
        x="mes_nombre",
        y=VALUE_COLUMN,
        order=month_order,
    )
    plt.title("Distribución mensual de exportaciones de flores naturales")
    plt.xlabel("Mes")
    plt.ylabel("Miles de dólares FOB")
    plt.xticks(rotation=45)

    save_current_plot(output_dir / "03_boxplot_mensual.svg")


def plot_histogram(data: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera el gráfico 04: histograma con curva de densidad.

    Parameters
    ----------
    data : pandas.DataFrame
        Base con la columna flores_naturales.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=VALUE_COLUMN, kde=True)
    plt.title("Distribución de la variable exportaciones de flores naturales")
    plt.xlabel("Miles de dólares FOB")
    plt.ylabel("Frecuencia")

    save_current_plot(output_dir / "04_histograma.svg")


def plot_year_month_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera el gráfico 05: heatmap de exportaciones por año y mes.

    Parameters
    ----------
    data : pandas.DataFrame
        Base con año, mes y flores_naturales.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    heatmap_table = data.pivot_table(
        index="anio",
        columns="mes",
        values=VALUE_COLUMN,
        aggfunc="mean",
    )

    plt.figure(figsize=(14, 9))
    sns.heatmap(heatmap_table, annot=False, cmap="YlGnBu")
    plt.title("Heatmap de exportaciones por año y mes")
    plt.xlabel("Mes")
    plt.ylabel("Año")

    save_current_plot(output_dir / "05_heatmap_anio_mes.svg")


def generate_exploratory_plots(data: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera los gráficos exploratorios 02, 03, 04 y 05.

    Parameters
    ----------
    data : pandas.DataFrame
        Base preparada para gráficos.
    output_dir : pathlib.Path
        Carpeta donde se guardarán los gráficos.
    """
    plot_moving_average(data, output_dir)
    plot_monthly_boxplot(data, output_dir)
    plot_histogram(data, output_dir)
    plot_year_month_heatmap(data, output_dir)


def stationarity_tests(series: pd.Series, name: str = "Serie") -> pd.DataFrame:
    """
    Ejecuta pruebas de estacionariedad ADF, Phillips-Perron, KPSS y DF-GLS.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal a evaluar.
    name : str, optional
        Nombre descriptivo para imprimir en consola.

    Returns
    -------
    pandas.DataFrame
        Tabla con estadístico, p-valor y decisión al 5%.
    """
    clean_series = series.dropna()
    results = []

    adf_result = ADF(clean_series)
    results.append(
        {
            "test": "ADF",
            "estadistico": adf_result.stat,
            "pvalor": adf_result.pvalue,
            "decision_5%": (
                "Estacionaria"
                if adf_result.pvalue < 0.05
                else "No estacionaria"
            ),
        }
    )

    pp_result = PhillipsPerron(clean_series)
    results.append(
        {
            "test": "Phillips-Perron",
            "estadistico": pp_result.stat,
            "pvalor": pp_result.pvalue,
            "decision_5%": (
                "Estacionaria"
                if pp_result.pvalue < 0.05
                else "No estacionaria"
            ),
        }
    )

    kpss_result = KPSS(clean_series)
    results.append(
        {
            "test": "KPSS",
            "estadistico": kpss_result.stat,
            "pvalor": kpss_result.pvalue,
            "decision_5%": (
                "Estacionaria"
                if kpss_result.pvalue > 0.05
                else "No estacionaria"
            ),
        }
    )

    dfgls_result = DFGLS(clean_series)
    results.append(
        {
            "test": "DF-GLS",
            "estadistico": dfgls_result.stat,
            "pvalor": dfgls_result.pvalue,
            "decision_5%": (
                "Estacionaria"
                if dfgls_result.pvalue < 0.05
                else "No estacionaria"
            ),
        }
    )

    results_df = pd.DataFrame(results)

    print(f"\nPruebas de estacionariedad - {name}")
    print(results_df.to_string(index=False))

    return results_df


def apply_differences(
    series: pd.Series,
    d: int = 0,
    seasonal_d: int = 0,
    period: int = 12,
) -> pd.Series:
    """
    Aplica diferencias regulares y estacionales a una serie temporal.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal original.
    d : int, optional
        Número de diferencias regulares.
    seasonal_d : int, optional
        Número de diferencias estacionales.
    period : int, optional
        Periodicidad estacional.

    Returns
    -------
    pandas.Series
        Serie diferenciada sin valores nulos.
    """
    transformed = series.copy()

    for _ in range(d):
        transformed = transformed.diff()

    for _ in range(seasonal_d):
        transformed = transformed.diff(period)

    return transformed.dropna()


def plot_decomposition(series: pd.Series, output_dir: Path) -> None:
    """
    Genera y guarda la descomposición estacional aditiva.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    decomposition = seasonal_decompose(series, model="additive", period=12)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    save_figure(fig, output_dir / "06_descomposicion_estacional.svg")


def plot_acf_pacf_graphs(series: pd.Series, output_dir: Path) -> None:
    """
    Genera y guarda los gráficos ACF y PACF de la serie original.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))

    plot_acf(series.dropna(), lags=48, ax=axes[0])
    axes[0].set_title("ACF - Serie original")

    plot_pacf(series.dropna(), lags=48, ax=axes[1], method="ywm")
    axes[1].set_title("PACF - Serie original")

    save_figure(fig, output_dir / "07_acf_pacf_serie_original.svg")

def plot_acf_pacf_graphs_t(t_series: pd.Series, output_dir: Path) -> None:
    """
    Genera y guarda los gráficos ACF y PACF de la serie original.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))

    plot_acf(t_series.dropna(), lags=48, ax=axes[0])
    axes[0].set_title("ACF - Serie transformada")

    plot_pacf(t_series.dropna(), lags=48, ax=axes[1], method="ywm")
    axes[1].set_title("PACF - Serie transformada")

    save_figure(fig, output_dir / "07_1_acf_pacf_serie_transformada.svg")


def kruskal_seasonality_test(series: pd.Series) -> dict:
    """
    Ejecuta la prueba de Kruskal-Wallis para estacionalidad mensual.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.

    Returns
    -------
    dict
        Estadístico, p-valor y conclusión de la prueba.
    """
    data = pd.DataFrame({"y": series})
    data["mes"] = data.index.month
    groups = [group["y"].dropna().values for _, group in data.groupby("mes")]

    statistic, pvalue = kruskal(*groups)
    conclusion = (
        "Hay evidencia de estacionalidad mensual."
        if pvalue < 0.05
        else "No hay evidencia suficiente de estacionalidad mensual."
    )

    print("\nKruskal-Wallis para estacionalidad mensual")
    print(f"Estadístico: {statistic}")
    print(f"p-valor    : {pvalue}")
    print(f"Conclusión : {conclusion}")

    return {
        "test": "Kruskal-Wallis",
        "estadistico": statistic,
        "pvalor": pvalue,
        "conclusion": conclusion,
    }


def fit_ets_models(series: pd.Series) -> tuple[dict, pd.DataFrame]:
    """
    Ajusta dos modelos ETS y genera una tabla comparativa.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.

    Returns
    -------
    tuple[dict, pandas.DataFrame]
        Diccionario de modelos ajustados y tabla comparativa.
    """
    train_series = series.iloc[-37:]

    ets_add_model = ETSModel(
        train_series,
        error="mul",
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        damped_trend=False,
    )
    ets_add_result = ets_add_model.fit(disp=False)

    ets_mul_model = ETSModel(
        train_series,
        error="mul",
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        damped_trend=False,
    )
    ets_mul_result = ets_mul_model.fit(disp=False)

    ets_results = {
        "ETS_M_A_A": ets_add_result,
        "ETS_M_A_M": ets_mul_result,
    }

    comparison = pd.DataFrame(
        {
            "Modelo": ["ETS(M,A,A)", "ETS(M,A,M)"],
            "AIC": [ets_add_result.aic, ets_mul_result.aic],
            "BIC": [ets_add_result.bic, ets_mul_result.bic],
            "LogLik": [ets_add_result.llf, ets_mul_result.llf],
            "MSE": [ets_add_result.mse, ets_mul_result.mse],
            "MAE": [ets_add_result.mae, ets_mul_result.mae],
        }
    ).sort_values("AIC")

    print("\n=== COMPARACIÓN ETS ===")
    print(comparison.to_string(index=False))

    return ets_results, comparison


def fit_prophet_models(
    series: pd.Series,
    output_dir: Path,
) -> tuple[dict, dict]:
    """
    Ajusta modelos Prophet aditivo y multiplicativo.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    output_dir : pathlib.Path
        Carpeta donde se guardarán gráficos Prophet.

    Returns
    -------
    tuple[dict, dict]
        Diccionario de modelos Prophet y diccionario de pronósticos.
    """
    prophet_data = series.reset_index()
    prophet_data.columns = ["ds", "y"]

    changepoints = [
        "2020-11-01",
        "2021-01-01",
        "2021-11-01",
        "2024-10-01",
    ]

    additive_model = Prophet(
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.2,
        changepoint_range=0.95,
        changepoints=changepoints,
    )
    additive_model.fit(prophet_data)

    future_additive = additive_model.make_future_dataframe(
        periods=24,
        freq="MS",
    )
    additive_forecast = additive_model.predict(future_additive)

    fig = additive_model.plot(additive_forecast)
    plt.title("Prophet aditivo - pronóstico")
    save_figure(fig, output_dir / "08_prophet_aditivo_pronostico.svg")

    fig = additive_model.plot_components(additive_forecast)
    save_figure(fig, output_dir / "09_prophet_aditivo_componentes.svg")

    multiplicative_model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.2,
        changepoint_range=0.95,
        changepoints=changepoints,
    )
    multiplicative_model.fit(prophet_data)

    future_multiplicative = multiplicative_model.make_future_dataframe(
        periods=24,
        freq="MS",
    )
    multiplicative_forecast = multiplicative_model.predict(
        future_multiplicative
    )

    fig = multiplicative_model.plot(multiplicative_forecast)
    plt.title("Prophet multiplicativo - pronóstico")
    save_figure(fig, output_dir / "10_prophet_multiplicativo_pronostico.svg")

    fig = multiplicative_model.plot_components(multiplicative_forecast)
    save_figure(fig, output_dir / "11_prophet_multiplicativo_componentes.svg")

    models = {
        "Prophet_Aditivo": additive_model,
        "Prophet_Multiplicativo": multiplicative_model,
    }

    forecasts = {
        "Prophet_Aditivo": additive_forecast,
        "Prophet_Multiplicativo": multiplicative_forecast,
    }

    return models, forecasts


def plot_prophet_comparison(
    series: pd.Series,
    prophet_forecasts: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compara valores reales contra Prophet aditivo y multiplicativo.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    prophet_forecasts : dict
        Diccionario con pronósticos Prophet.
    output_dir : pathlib.Path
        Carpeta donde se guardará el gráfico.

    Returns
    -------
    pandas.DataFrame
        Tabla de comparación entre valores reales y predicciones Prophet.
    """
    idx_eval = pd.date_range(
        start="2021-01-01",
        end="2026-12-01",
        freq="MS",
    )

    comparison = pd.DataFrame(
        {
            "Real": series.reindex(idx_eval),
            "Prophet_add": (
                prophet_forecasts["Prophet_Aditivo"]
                .set_index("ds")["yhat"]
                .reindex(idx_eval)
            ),
            "Prophet_mul": (
                prophet_forecasts["Prophet_Multiplicativo"]
                .set_index("ds")["yhat"]
                .reindex(idx_eval)
            ),
        }
    )

    plt.figure(figsize=(14, 6))
    plt.plot(
        comparison.index,
        comparison["Real"],
        marker="o",
        linewidth=2,
        label="Valores reales",
    )
    plt.plot(
        comparison.index,
        comparison["Prophet_add"],
        linewidth=2,
        label="Prophet aditivo",
    )
    plt.plot(
        comparison.index,
        comparison["Prophet_mul"],
        linewidth=2,
        label="Prophet multiplicativo",
    )
    plt.title("Valores reales vs Prophet")
    plt.xlabel("Fecha")
    plt.ylabel("Flores naturales")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_current_plot(output_dir / "12_comparacion_prophet.svg")

    print("\n=== COMPARACIÓN PROPHET ===")
    print(comparison.to_string())

    return comparison


def fit_custom_sarimax(
    series: pd.Series,
    ar_lags: list[int],
    ma_lags: list[int],
    sar_lags: list[int],
    sma_lags: list[int],
    d: int = 1,
    seasonal_d: int = 0,
    seasonal_period: int = 12,
    trend: str = "c",
):
    """
    Ajusta un modelo SARIMAX con rezagos específicos.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    ar_lags : list[int]
        Rezagos autorregresivos no estacionales.
    ma_lags : list[int]
        Rezagos de media móvil no estacionales.
    sar_lags : list[int]
        Rezagos autorregresivos estacionales.
    sma_lags : list[int]
        Rezagos de media móvil estacionales.
    d : int, optional
        Orden de diferenciación regular.
    seasonal_d : int, optional
        Orden de diferenciación estacional.
    seasonal_period : int, optional
        Periodicidad estacional.
    trend : str, optional
        Tipo de tendencia usada por SARIMAX.

    Returns
    -------
    statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
        Resultado ajustado del modelo SARIMAX.
    """
    model = SARIMAX(
        series,
        order=(
            ar_lags if len(ar_lags) > 0 else 0,
            d,
            ma_lags if len(ma_lags) > 0 else 0,
        ),
        seasonal_order=(
            sar_lags if len(sar_lags) > 0 else 0,
            seasonal_d,
            sma_lags if len(sma_lags) > 0 else 0,
            seasonal_period,
        ),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    result = model.fit(disp=False)

    return result


def fit_sarimax_collection(
    series: pd.Series,
    model_specs: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Ajusta una colección de modelos SARIMAX y resume sus resultados.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    model_specs : dict
        Diccionario de especificaciones SARIMAX.

    Returns
    -------
    tuple[dict, pandas.DataFrame, pandas.DataFrame]
        Resultados ajustados, resumen de modelos y tabla de coeficientes.
    """
    results = {}
    summary_rows = []
    coefficient_rows = []

    for model_name, spec in model_specs.items():
        try:
            result = fit_custom_sarimax(
                series=series,
                ar_lags=spec["AR"],
                ma_lags=spec["MA"],
                sar_lags=spec["SAR"],
                sma_lags=spec["SMA"],
                d=1,
                seasonal_d=0,
                seasonal_period=12,
                trend="c",
            )

            results[model_name] = result

            summary_rows.append(
                {
                    "Model": model_name,
                    "AR": spec["AR"],
                    "MA": spec["MA"],
                    "SAR": spec["SAR"],
                    "SMA": spec["SMA"],
                    "AIC": result.aic,
                    "BIC": result.bic,
                    "HQIC": getattr(result, "hqic", np.nan),
                    "LogLik": result.llf,
                    "nobs": result.nobs,
                    "Converged": getattr(
                        result,
                        "mle_retvals",
                        {},
                    ).get("converged", True),
                }
            )

            coefficients = pd.DataFrame(
                {
                    "Model": model_name,
                    "Parameter": result.params.index,
                    "Coefficient": result.params.values,
                    "Std.Error": result.bse.values,
                    "z": result.zvalues.values,
                    "p_value": result.pvalues.values,
                }
            )
            coefficient_rows.append(coefficients)

            print(f"{model_name}: OK")

        except Exception as error:
            summary_rows.append(
                {
                    "Model": model_name,
                    "AR": spec["AR"],
                    "MA": spec["MA"],
                    "SAR": spec["SAR"],
                    "SMA": spec["SMA"],
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "HQIC": np.nan,
                    "LogLik": np.nan,
                    "nobs": np.nan,
                    "Converged": False,
                    "Error": str(error),
                }
            )
            print(f"{model_name}: ERROR -> {error}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="AIC",
        na_position="last",
    )
    coefficients_df = (
        pd.concat(coefficient_rows, ignore_index=True)
        if coefficient_rows
        else pd.DataFrame()
    )

    print("\n=== RESUMEN DE MODELOS SARIMAX ===")
    print(summary_df.to_string(index=False))

    print("\n=== COEFICIENTES SARIMAX ===")
    print(coefficients_df.to_string(index=False))

    return results, summary_df, coefficients_df


def diagnose_garch_need(sarimax_results: dict) -> pd.DataFrame:
    """
    Diagnostica si los residuos SARIMAX justifican un modelo GARCH.

    Parameters
    ----------
    sarimax_results : dict
        Diccionario de resultados SARIMAX ajustados.

    Returns
    -------
    pandas.DataFrame
        Tabla con pruebas ARCH-LM y Ljung-Box.
    """
    diagnostic_rows = []

    for model_name, result in sarimax_results.items():
        residuals = pd.Series(result.resid).dropna()
        residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()

        arch_lm_stat, arch_lm_pvalue, arch_f_stat, arch_f_pvalue = het_arch(
            residuals,
            nlags=12,
        )

        ljung_box_resid = acorr_ljungbox(
            residuals,
            lags=[12],
            return_df=True,
        )
        ljung_box_resid_squared = acorr_ljungbox(
            residuals**2,
            lags=[12],
            return_df=True,
        )

        diagnostic_rows.append(
            {
                "Modelo": model_name,
                "ARCH_LM_stat": arch_lm_stat,
                "ARCH_LM_pvalue": arch_lm_pvalue,
                "ARCH_F_stat": arch_f_stat,
                "ARCH_F_pvalue": arch_f_pvalue,
                "LjungBox_resid_pvalue_lag12": (
                    ljung_box_resid["lb_pvalue"].iloc[0]
                ),
                "LjungBox_resid2_pvalue_lag12": (
                    ljung_box_resid_squared["lb_pvalue"].iloc[0]
                ),
                "Decision_ARCH_5%": (
                    "Justifica GARCH"
                    if arch_lm_pvalue < 0.05
                    else "No justifica GARCH"
                ),
            }
        )

    diagnostics_df = pd.DataFrame(diagnostic_rows)

    print("\n=== DIAGNÓSTICO PARA JUSTIFICAR GARCH ===")
    print(diagnostics_df.to_string(index=False))

    return diagnostics_df


def fit_sarimax_garch(
    series: pd.Series,
    ar_lags: list[int],
    ma_lags: list[int],
    sar_lags: list[int],
    sma_lags: list[int],
    d: int = 1,
    seasonal_d: int = 0,
    seasonal_period: int = 12,
    trend: str = "c",
    garch_p: int = 1,
    garch_q: int = 1,
    garch_dist: str = "normal",
):
    """
    Ajusta SARIMAX para la media y GARCH(1,1) sobre los residuos.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    ar_lags : list[int]
        Rezagos autorregresivos no estacionales.
    ma_lags : list[int]
        Rezagos de media móvil no estacionales.
    sar_lags : list[int]
        Rezagos autorregresivos estacionales.
    sma_lags : list[int]
        Rezagos de media móvil estacionales.
    d : int, optional
        Diferenciación regular.
    seasonal_d : int, optional
        Diferenciación estacional.
    seasonal_period : int, optional
        Periodicidad estacional.
    trend : str, optional
        Tipo de tendencia para SARIMAX.
    garch_p : int, optional
        Orden ARCH del GARCH.
    garch_q : int, optional
        Orden GARCH del GARCH.
    garch_dist : str, optional
        Distribución de errores del modelo GARCH.

    Returns
    -------
    tuple
        Resultado SARIMAX y resultado GARCH.
    """
    sarimax_result = fit_custom_sarimax(
        series=series,
        ar_lags=ar_lags,
        ma_lags=ma_lags,
        sar_lags=sar_lags,
        sma_lags=sma_lags,
        d=d,
        seasonal_d=seasonal_d,
        seasonal_period=seasonal_period,
        trend=trend,
    )

    residuals = sarimax_result.resid.dropna()
    residuals = residuals.replace([np.inf, -np.inf], np.nan).dropna()

    garch_model = arch_model(
        residuals,
        mean="Zero",
        vol="GARCH",
        p=garch_p,
        q=garch_q,
        dist=garch_dist,
        rescale=True,
    )
    garch_result = garch_model.fit(disp="off")

    return sarimax_result, garch_result


def fit_sarimax_garch_collection(
    series: pd.Series,
    model_specs: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Ajusta modelos SARIMAX + GARCH(1,1) para una colección de modelos.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    model_specs : dict
        Diccionario con especificaciones SARIMAX.

    Returns
    -------
    tuple[dict, pandas.DataFrame, pandas.DataFrame]
        Resultados ajustados, resumen comparativo y coeficientes.
    """
    results = {}
    summary_rows = []
    sarimax_coefficient_rows = []
    garch_coefficient_rows = []

    for model_name, spec in model_specs.items():
        combined_name = f"{model_name} + GARCH(1,1)"

        try:
            sarimax_result, garch_result = fit_sarimax_garch(
                series=series,
                ar_lags=spec["AR"],
                ma_lags=spec["MA"],
                sar_lags=spec["SAR"],
                sma_lags=spec["SMA"],
                d=1,
                seasonal_d=0,
                seasonal_period=12,
                trend="c",
                garch_p=1,
                garch_q=1,
                garch_dist="normal",
            )

            results[combined_name] = {
                "sarimax": sarimax_result,
                "garch": garch_result,
            }

            loglik_combined = (
                sarimax_result.llf + garch_result.loglikelihood
            )
            n_params_total = len(sarimax_result.params) + len(
                garch_result.params
            )

            aic_combined = -2 * loglik_combined + 2 * n_params_total
            bic_combined = (
                -2 * loglik_combined
                + np.log(len(series)) * n_params_total
            )

            summary_rows.append(
                {
                    "Model": combined_name,
                    "AR": spec["AR"],
                    "MA": spec["MA"],
                    "SAR": spec["SAR"],
                    "SMA": spec["SMA"],
                    "SARIMAX_AIC": sarimax_result.aic,
                    "SARIMAX_BIC": sarimax_result.bic,
                    "SARIMAX_LogLik": sarimax_result.llf,
                    "GARCH_AIC": garch_result.aic,
                    "GARCH_BIC": garch_result.bic,
                    "GARCH_LogLik": garch_result.loglikelihood,
                    "Combined_AIC_Approx": aic_combined,
                    "Combined_BIC_Approx": bic_combined,
                    "Combined_LogLik_Approx": loglik_combined,
                    "nobs_sarimax": sarimax_result.nobs,
                    "nobs_garch": garch_result.nobs,
                    "SARIMAX_Converged": getattr(
                        sarimax_result,
                        "mle_retvals",
                        {},
                    ).get("converged", True),
                    "GARCH_Converged": garch_result.convergence_flag == 0,
                }
            )

            sarimax_coefficients = pd.DataFrame(
                {
                    "Model": combined_name,
                    "Component": "SARIMAX",
                    "Parameter": sarimax_result.params.index,
                    "Coefficient": sarimax_result.params.values,
                    "Std.Error": sarimax_result.bse.values,
                    "z": sarimax_result.zvalues.values,
                    "p_value": sarimax_result.pvalues.values,
                }
            )
            sarimax_coefficient_rows.append(sarimax_coefficients)

            garch_coefficients = pd.DataFrame(
                {
                    "Model": combined_name,
                    "Component": "GARCH(1,1)",
                    "Parameter": garch_result.params.index,
                    "Coefficient": garch_result.params.values,
                    "Std.Error": garch_result.std_err.values,
                    "z": garch_result.tvalues.values,
                    "p_value": garch_result.pvalues.values,
                }
            )
            garch_coefficient_rows.append(garch_coefficients)

            print(f"{combined_name}: OK")

        except Exception as error:
            summary_rows.append(
                {
                    "Model": combined_name,
                    "AR": spec["AR"],
                    "MA": spec["MA"],
                    "SAR": spec["SAR"],
                    "SMA": spec["SMA"],
                    "SARIMAX_AIC": np.nan,
                    "SARIMAX_BIC": np.nan,
                    "SARIMAX_LogLik": np.nan,
                    "GARCH_AIC": np.nan,
                    "GARCH_BIC": np.nan,
                    "GARCH_LogLik": np.nan,
                    "Combined_AIC_Approx": np.nan,
                    "Combined_BIC_Approx": np.nan,
                    "Combined_LogLik_Approx": np.nan,
                    "nobs_sarimax": np.nan,
                    "nobs_garch": np.nan,
                    "SARIMAX_Converged": False,
                    "GARCH_Converged": False,
                    "Error": str(error),
                }
            )
            print(f"{combined_name}: ERROR -> {error}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="Combined_AIC_Approx",
        na_position="last",
    )

    sarimax_coefficients_df = (
        pd.concat(sarimax_coefficient_rows, ignore_index=True)
        if sarimax_coefficient_rows
        else pd.DataFrame()
    )
    garch_coefficients_df = (
        pd.concat(garch_coefficient_rows, ignore_index=True)
        if garch_coefficient_rows
        else pd.DataFrame()
    )

    coefficients_df = (
        pd.concat(
            [sarimax_coefficients_df, garch_coefficients_df],
            ignore_index=True,
        )
        if not sarimax_coefficients_df.empty
        or not garch_coefficients_df.empty
        else pd.DataFrame()
    )

    print("\n=== RESUMEN DE MODELOS SARIMAX + GARCH(1,1) ===")
    print(summary_df.to_string(index=False))

    print("\n=== COEFICIENTES SARIMAX + GARCH(1,1) ===")
    print(coefficients_df.to_string(index=False))

    return results, summary_df, coefficients_df


def calculate_metrics(y_real: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calcula MAE, RMSE y MAPE eliminando valores nulos.

    Parameters
    ----------
    y_real : pandas.Series
        Valores reales.
    y_pred : pandas.Series
        Valores predichos.

    Returns
    -------
    dict
        Métricas de evaluación y número de observaciones evaluadas.
    """
    eval_data = pd.DataFrame(
        {
            "Real": y_real,
            "Prediccion": y_pred,
        }
    ).dropna()

    if eval_data.empty:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_eval": 0,
        }

    mae = mean_absolute_error(eval_data["Real"], eval_data["Prediccion"])
    rmse = np.sqrt(
        mean_squared_error(eval_data["Real"], eval_data["Prediccion"])
    )

    mape_data = eval_data[eval_data["Real"] != 0].copy()

    if not mape_data.empty:
        mape = (
            np.mean(
                np.abs(
                    (mape_data["Real"] - mape_data["Prediccion"])
                    / mape_data["Real"]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "n_eval": len(eval_data),
    }


def build_in_sample_predictions(
    series: pd.Series,
    ets_results: dict,
    prophet_forecasts: dict,
    sarimax_garch_results: dict,
) -> tuple[dict, dict]:
    """
    Construye predicciones dentro de muestra para comparar modelos.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual.
    ets_results : dict
        Modelos ETS ajustados.
    prophet_forecasts : dict
        Pronósticos Prophet.
    sarimax_garch_results : dict
        Modelos SARIMAX + GARCH ajustados.

    Returns
    -------
    tuple[dict, dict]
        Predicciones por modelo y volatilidades condicionales GARCH.
    """
    predictions = {
        "ETS_M_A_A": ets_results["ETS_M_A_A"].fittedvalues,
        "ETS_M_A_M": ets_results["ETS_M_A_M"].fittedvalues,
        "Prophet_Aditivo": (
            prophet_forecasts["Prophet_Aditivo"].set_index("ds")["yhat"]
        ),
        "Prophet_Multiplicativo": (
            prophet_forecasts["Prophet_Multiplicativo"].set_index("ds")[
                "yhat"
            ]
        ),
    }
    garch_volatilities = {}

    for model_name, model_objects in sarimax_garch_results.items():
        sarimax_result = model_objects["sarimax"]
        garch_result = model_objects["garch"]

        predicted_mean = sarimax_result.get_prediction(
            start=series.index.min(),
            end=series.index.max(),
            dynamic=False,
        ).predicted_mean

        clean_name = f"SARIMAX_GARCH_{model_name}"
        predictions[clean_name] = predicted_mean
        garch_volatilities[clean_name] = garch_result.conditional_volatility

    return predictions, garch_volatilities


def compare_predictions_for_period(
    series: pd.Series,
    predictions: dict,
    period_index: pd.DatetimeIndex,
    period_name: str,
) -> pd.DataFrame:
    """
    Compara modelos para un periodo específico.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual con valores reales.
    predictions : dict
        Diccionario de predicciones por modelo.
    period_index : pandas.DatetimeIndex
        Fechas a evaluar.
    period_name : str
        Nombre del periodo evaluado.

    Returns
    -------
    pandas.DataFrame
        Tabla comparativa con MAE, RMSE, MAPE y n_eval.
    """
    real_values = series.reindex(period_index)
    rows = []

    for model_name, prediction in predictions.items():
        metrics = calculate_metrics(
            real_values,
            prediction.reindex(period_index),
        )
        rows.append(
            {
                "Periodo": period_name,
                "Modelo": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE (%)": metrics["MAPE"],
                "n_eval": metrics["n_eval"],
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by="RMSE",
        ascending=True,
    )

    print(f"\n=== COMPARACIÓN DE MODELOS - {period_name.upper()} ===")
    print(comparison_df.to_string(index=False))

    return comparison_df


def build_real_vs_predictions_table(
    series: pd.Series,
    predictions: dict,
    period_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Construye una tabla con valores reales y predicciones por modelo.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal mensual con valores reales.
    predictions : dict
        Diccionario de predicciones por modelo.
    period_index : pandas.DatetimeIndex
        Fechas a incluir en la comparación.

    Returns
    -------
    pandas.DataFrame
        Tabla con valores reales y predicciones.
    """
    comparison = pd.DataFrame({"Real": series.reindex(period_index)})

    for model_name, prediction in predictions.items():
        comparison[model_name] = prediction.reindex(period_index)

    return comparison


def build_garch_volatility_table(
    garch_volatilities: dict,
    period_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Construye una tabla de volatilidades condicionales GARCH.

    Parameters
    ----------
    garch_volatilities : dict
        Diccionario de volatilidades por modelo.
    period_index : pandas.DatetimeIndex
        Fechas a incluir.

    Returns
    -------
    pandas.DataFrame
        Tabla de volatilidades condicionales.
    """
    volatility = pd.DataFrame(index=period_index)

    for model_name, values in garch_volatilities.items():
        volatility[model_name] = values.reindex(period_index)

    return volatility


def forecast_new_data_period(
    new_data: pd.Series,
    ets_results: dict,
    prophet_models: dict,
    sarimax_garch_results: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera predicciones y métricas para los nuevos datos disponibles.

    Parameters
    ----------
    new_data : pandas.Series
        Serie con nuevos valores reales fuera de la muestra original.
    ets_results : dict
        Modelos ETS ajustados.
    prophet_models : dict
        Modelos Prophet ajustados.
    sarimax_garch_results : dict
        Modelos SARIMAX + GARCH ajustados.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Tabla de métricas y tabla real vs predicciones.
    """
    new_index = new_data.index
    steps_forecast = len(new_index)
    predictions = {}

    ets_add_forecast = ets_results["ETS_M_A_A"].forecast(steps=steps_forecast)
    ets_add_forecast.index = new_index
    predictions["ETS_M_A_A"] = ets_add_forecast

    ets_mul_forecast = ets_results["ETS_M_A_M"].forecast(steps=steps_forecast)
    ets_mul_forecast.index = new_index
    predictions["ETS_M_A_M"] = ets_mul_forecast

    for model_name, model_objects in sarimax_garch_results.items():
        sarimax_result = model_objects["sarimax"]
        sarimax_forecast = sarimax_result.get_forecast(
            steps=steps_forecast
        ).predicted_mean
        sarimax_forecast.index = new_index

        clean_name = f"SARIMAX_GARCH_{model_name}"
        predictions[clean_name] = sarimax_forecast

    future_additive = prophet_models["Prophet_Aditivo"].make_future_dataframe(
        periods=steps_forecast,
        freq="MS",
    )
    additive_forecast = prophet_models["Prophet_Aditivo"].predict(
        future_additive
    )
    predictions["Prophet_Aditivo"] = (
        additive_forecast.set_index("ds")["yhat"].reindex(new_index)
    )

    future_multiplicative = prophet_models[
        "Prophet_Multiplicativo"
    ].make_future_dataframe(periods=steps_forecast, freq="MS")
    multiplicative_forecast = prophet_models[
        "Prophet_Multiplicativo"
    ].predict(future_multiplicative)
    predictions["Prophet_Multiplicativo"] = (
        multiplicative_forecast.set_index("ds")["yhat"].reindex(new_index)
    )

    metric_rows = []

    for model_name, prediction in predictions.items():
        metrics = calculate_metrics(new_data, prediction)
        metric_rows.append(
            {
                "Periodo": "Febrero-Marzo 2026",
                "Modelo": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE (%)": metrics["MAPE"],
                "n_eval": metrics["n_eval"],
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by="RMSE",
        ascending=True,
    )

    comparison_df = pd.DataFrame({"Real": new_data})

    for model_name, prediction in predictions.items():
        comparison_df[model_name] = prediction.reindex(new_index)

    print("\n=== COMPARACIÓN DE MODELOS - FEBRERO Y MARZO 2026 ===")
    print(metrics_df.to_string(index=False))

    print("\n=== VALORES REALES VS PREDICCIONES - FEBRERO Y MARZO 2026 ===")
    print(comparison_df.to_string())

    return metrics_df, comparison_df


def export_results(
    output_dir: Path,
    stationarity_df: pd.DataFrame,
    ets_comparison: pd.DataFrame,
    sarimax_summary: pd.DataFrame,
    sarimax_coefficients: pd.DataFrame,
    selected_sarimax_summary: pd.DataFrame,
    selected_sarimax_coefficients: pd.DataFrame,
    garch_diagnostics: pd.DataFrame,
    sarimax_garch_summary: pd.DataFrame,
    sarimax_garch_coefficients: pd.DataFrame,
    model_comparison_2025: pd.DataFrame,
    model_comparison_january_2025: pd.DataFrame,
    real_vs_predictions_2025: pd.DataFrame,
    garch_volatility_2025: pd.DataFrame,
    new_data_metrics: pd.DataFrame,
    new_data_comparison: pd.DataFrame,
    kruskal_result: dict,
) -> None:
    """
    Exporta tablas de resultados a archivos Excel.

    Parameters
    ----------
    output_dir : pathlib.Path
        Carpeta de salida.
    stationarity_df : pandas.DataFrame
        Resultado de pruebas de estacionariedad.
    ets_comparison : pandas.DataFrame
        Comparación de modelos ETS.
    sarimax_summary : pandas.DataFrame
        Resumen de todos los modelos SARIMAX.
    sarimax_coefficients : pandas.DataFrame
        Coeficientes de todos los modelos SARIMAX.
    selected_sarimax_summary : pandas.DataFrame
        Resumen de modelos SARIMAX seleccionados.
    selected_sarimax_coefficients : pandas.DataFrame
        Coeficientes de modelos SARIMAX seleccionados.
    garch_diagnostics : pandas.DataFrame
        Diagnóstico ARCH/GARCH.
    sarimax_garch_summary : pandas.DataFrame
        Resumen SARIMAX + GARCH.
    sarimax_garch_coefficients : pandas.DataFrame
        Coeficientes SARIMAX + GARCH.
    model_comparison_2025 : pandas.DataFrame
        Métricas para el año 2025.
    model_comparison_january_2025 : pandas.DataFrame
        Métricas para enero de 2025.
    real_vs_predictions_2025 : pandas.DataFrame
        Valores reales vs predicciones para 2025.
    garch_volatility_2025 : pandas.DataFrame
        Volatilidades condicionales GARCH para 2025.
    new_data_metrics : pandas.DataFrame
        Métricas para nuevos datos de 2026.
    new_data_comparison : pandas.DataFrame
        Valores reales vs predicciones para nuevos datos.
    kruskal_result : dict
        Resultado de la prueba Kruskal-Wallis.
    """
    stationarity_path = output_dir / "pruebas_estadisticas.xlsx"
    models_path = output_dir / "resultados_modelos.xlsx"
    predictions_path = output_dir / "comparacion_predicciones.xlsx"

    with pd.ExcelWriter(stationarity_path, engine="openpyxl") as writer:
        stationarity_df.to_excel(
            writer,
            sheet_name="Estacionariedad",
            index=False,
        )
        pd.DataFrame([kruskal_result]).to_excel(
            writer,
            sheet_name="Kruskal",
            index=False,
        )

    with pd.ExcelWriter(models_path, engine="openpyxl") as writer:
        ets_comparison.to_excel(writer, sheet_name="ETS", index=False)
        sarimax_summary.to_excel(
            writer,
            sheet_name="SARIMAX_Resumen",
            index=False,
        )
        sarimax_coefficients.to_excel(
            writer,
            sheet_name="SARIMAX_Coef",
            index=False,
        )
        selected_sarimax_summary.to_excel(
            writer,
            sheet_name="SARIMAX_Selected",
            index=False,
        )
        selected_sarimax_coefficients.to_excel(
            writer,
            sheet_name="SARIMAX_Selected_Coef",
            index=False,
        )
        garch_diagnostics.to_excel(
            writer,
            sheet_name="Diagnostico_GARCH",
            index=False,
        )
        sarimax_garch_summary.to_excel(
            writer,
            sheet_name="SARIMAX_GARCH",
            index=False,
        )
        sarimax_garch_coefficients.to_excel(
            writer,
            sheet_name="SARIMAX_GARCH_Coef",
            index=False,
        )

    with pd.ExcelWriter(predictions_path, engine="openpyxl") as writer:
        model_comparison_2025.to_excel(
            writer,
            sheet_name="Metricas_2025",
            index=False,
        )
        model_comparison_january_2025.to_excel(
            writer,
            sheet_name="Metricas_Ene_2025",
            index=False,
        )
        real_vs_predictions_2025.to_excel(
            writer,
            sheet_name="Real_vs_Pred_2025",
        )
        garch_volatility_2025.to_excel(
            writer,
            sheet_name="Volatilidad_GARCH_2025",
        )
        new_data_metrics.to_excel(
            writer,
            sheet_name="Metricas_2026_new",
            index=False,
        )
        new_data_comparison.to_excel(
            writer,
            sheet_name="Real_vs_Pred_2026_new",
        )

    print("\nArchivos Excel guardados:")
    print(f"- {stationarity_path.resolve()}")
    print(f"- {models_path.resolve()}")
    print(f"- {predictions_path.resolve()}")


def run_analysis() -> None:
    """
    Ejecuta el flujo completo de análisis, gráficos, modelos y exportación.
    """
    configure_environment()

    raw_data = load_data(INPUT_FILE)
    print_data_quality(raw_data)

    clean_data = clean_data_for_plots(raw_data)
    plot_data = add_plot_columns(clean_data)
    monthly_series = build_monthly_series(clean_data)
    new_data = pd.Series(NEW_DATA_VALUES, name="Nuevos")

    print("\nObservaciones:", len(monthly_series))
    print(
        "Periodo:",
        monthly_series.index.min(),
        "a",
        monthly_series.index.max(),
    )

    generate_exploratory_plots(plot_data, OUTPUT_DIR)
    y_transf = apply_differences(monthly_series, d=1, seasonal_d=0, period=12)

    stationarity_df = stationarity_tests(
        monthly_series,
        name="Serie original",
    )
    
    plot_decomposition(monthly_series, OUTPUT_DIR)
    plot_acf_pacf_graphs(monthly_series, OUTPUT_DIR)
    plot_acf_pacf_graphs_t(y_transf, OUTPUT_DIR)
    kruskal_result = kruskal_seasonality_test(monthly_series)

    ets_results, ets_comparison = fit_ets_models(monthly_series)
    prophet_models, prophet_forecasts = fit_prophet_models(
        monthly_series,
        OUTPUT_DIR,
    )
    plot_prophet_comparison(
        monthly_series,
        prophet_forecasts,
        OUTPUT_DIR,
    )

    sarimax_results, sarimax_summary, sarimax_coefficients = (
        fit_sarimax_collection(monthly_series, ALL_SARIMA_MODELS)
    )
    selected_sarimax_results, selected_sarimax_summary, (
        selected_sarimax_coefficients
    ) = fit_sarimax_collection(monthly_series, SELECTED_SARIMA_MODELS)

    garch_diagnostics = diagnose_garch_need(selected_sarimax_results)

    sarimax_garch_results, sarimax_garch_summary, sarimax_garch_coefficients = (
        fit_sarimax_garch_collection(monthly_series, SELECTED_SARIMA_MODELS)
    )

    predictions, garch_volatilities = build_in_sample_predictions(
        monthly_series,
        ets_results,
        prophet_forecasts,
        sarimax_garch_results,
    )

    idx_2025 = pd.date_range(
        start="2025-01-01",
        end="2025-12-01",
        freq="MS",
    )
    idx_january_2025 = pd.date_range(
        start="2025-01-01",
        end="2025-01-01",
        freq="MS",
    )

    model_comparison_2025 = compare_predictions_for_period(
        monthly_series,
        predictions,
        idx_2025,
        "Año 2025",
    )
    model_comparison_january_2025 = compare_predictions_for_period(
        monthly_series,
        predictions,
        idx_january_2025,
        "Enero 2025",
    )

    real_vs_predictions_2025 = build_real_vs_predictions_table(
        monthly_series,
        predictions,
        idx_2025,
    )
    garch_volatility_2025 = build_garch_volatility_table(
        garch_volatilities,
        idx_2025,
    )

    print("\n=== VALORES REALES VS PREDICCIONES - 2025 ===")
    print(real_vs_predictions_2025.to_string())

    print("\n=== VOLATILIDAD CONDICIONAL GARCH - 2025 ===")
    print(garch_volatility_2025.to_string())

    new_data_metrics, new_data_comparison = forecast_new_data_period(
        new_data,
        ets_results,
        prophet_models,
        sarimax_garch_results,
    )

    export_results(
        output_dir=OUTPUT_DIR,
        stationarity_df=stationarity_df,
        ets_comparison=ets_comparison,
        sarimax_summary=sarimax_summary,
        sarimax_coefficients=sarimax_coefficients,
        selected_sarimax_summary=selected_sarimax_summary,
        selected_sarimax_coefficients=selected_sarimax_coefficients,
        garch_diagnostics=garch_diagnostics,
        sarimax_garch_summary=sarimax_garch_summary,
        sarimax_garch_coefficients=sarimax_garch_coefficients,
        model_comparison_2025=model_comparison_2025,
        model_comparison_january_2025=model_comparison_january_2025,
        real_vs_predictions_2025=real_vs_predictions_2025,
        garch_volatility_2025=garch_volatility_2025,
        new_data_metrics=new_data_metrics,
        new_data_comparison=new_data_comparison,
        kruskal_result=kruskal_result,
    )

    print("\nGráficos y resultados generados correctamente.")
    print(f"Carpeta de salida: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_analysis()
