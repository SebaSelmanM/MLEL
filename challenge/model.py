import pandas as pd
import numpy as np

from typing import Tuple, Union, List

# Modelos y utilidades
import xgboost as xgb
from datetime import datetime

class DelayModel:
    """
    Clase principal para entrenar y predecir retrasos.
    """

    def __init__(self):
        # El modelo final se guarda en este atributo
        self._model = None  
        # Lista de features seleccionadas (top 10) según el análisis
        self._top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        # Guardamos el factor de balance para usar el mismo en predict si fuera necesario
        # (XGBoost no lo requiere al predecir, pero a veces es útil tenerlo)
        self._scale_pos_weight = 1.0

    def _get_period_day(self, date_str: str) -> str:
        """
        Dada una fecha con formato '%Y-%m-%d %H:%M:%S',
        retorna 'mañana', 'tarde' o 'noche' según la hora.
        """
        date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()

        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        # Lo demás se asume 'noche': 19:00 a 04:59
        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        else:
            return 'noche'

    def _is_high_season(self, date_str: str) -> int:
        """
        Retorna 1 si la fecha cae en las ventanas consideradas 'alta temporada',
        de lo contrario 0.
        """
        fecha = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        año = fecha.year

        # Rangos (usamos 'replace(year=...)' para fijar el año)
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=año)

        if ((range1_min <= fecha <= range1_max) or
            (range2_min <= fecha <= range2_max) or
            (range3_min <= fecha <= range3_max) or
            (range4_min <= fecha <= range4_max)):
            return 1
        else:
            return 0

    def _get_min_diff(self, row: pd.Series) -> float:
        """
        Calcula la diferencia en minutos entre 'Fecha-O' y 'Fecha-I' de una fila.
        """
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return (fecha_o - fecha_i).total_seconds() / 60.0

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        1) Crea columnas period_day, high_season, min_diff, delay (si procede).
        2) Convierte a dummies las columnas relevantes.
        3) Retorna features (y target si 'target_column' está definido).

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target (si 'target_column' se especifica).
            or
            pd.DataFrame: features (si 'target_column' no se especifica).
        """

        # Para evitar modificar el dataframe original
        df = data.copy()

        # Generación de las columnas adicionales:
        if 'Fecha-I' in df.columns:
            df['period_day'] = df['Fecha-I'].apply(self._get_period_day)
            df['high_season'] = df['Fecha-I'].apply(self._is_high_season)

        if 'Fecha-O' in df.columns and 'Fecha-I' in df.columns:
            df['min_diff'] = df.apply(self._get_min_diff, axis=1)
            # Generamos la columna 'delay' (threshold de 15 minutos)
            df['delay'] = np.where(df['min_diff'] > 15, 1, 0)

        # Conviértete a variables dummies las columnas relevantes.
        # Ajusta según las columnas que se usaron en el notebook.
        # (OPERA, TIPOVUELO, MES, etc.) Asegúrate de que existan antes.
        feature_df = []

        if 'OPERA' in df.columns:
            feature_df.append(pd.get_dummies(df['OPERA'], prefix='OPERA'))
        if 'TIPOVUELO' in df.columns:
            feature_df.append(pd.get_dummies(df['TIPOVUELO'], prefix='TIPOVUELO'))
        if 'MES' in df.columns:
            feature_df.append(pd.get_dummies(df['MES'], prefix='MES'))

        # Concatenamos todo
        if feature_df:
            X = pd.concat(feature_df, axis=1)
        else:
            X = pd.DataFrame()

        # Si trabajas solo con las "top 10 features", podrías filtrar aquí
        # pero OJO: si filtras, asegúrate de no perder columnas en predict().
        # Ejemplo:
        # A) En entrenamiento:
        #    X = X.reindex(columns=self._top_10_features, fill_value=0)
        # B) En inferencia:
        #    Haces lo mismo, para garantizar que existan y el orden sea correcto.

        # Por seguridad, reindex con las top_10 para mantener orden y no faltar cols:
        X = X.reindex(columns=self._top_10_features, fill_value=0)

        # Retornamos target si corresponde
        if target_column and target_column in df.columns:
            y = df[[target_column]]
            return X, y
        else:
            return X

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed features.
            target (pd.DataFrame): target (delay).
        """
        # Calculamos scale_pos_weight para manejar desbalance
        # (Número de 0 / número de 1)
        # Flatten the target DataFrame to a Series
        target_series = target.squeeze()

        # Ensure target_series is a Series
        if isinstance(target_series, pd.DataFrame):
            target_series = target_series.iloc[:, 0]

        n_delay_0 = len(target_series[target_series == 0])
        n_delay_1 = len(target_series[target_series == 1])
        self._scale_pos_weight = n_delay_0 / n_delay_1 if n_delay_1 else 1.0

        # Entrenamos XGBoost (hiperparámetros básicos)
        self._model = xgb.XGBClassifier(
            random_state=42,
            learning_rate=0.01,
            scale_pos_weight=self._scale_pos_weight
        )
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed features.
        
        Returns:
            (List[int]): predicted targets (0 / 1).
        """
        # Asegurarse que se haya realizado el entrenamiento
        if self._model is None:
            return [0] * len(features)  #dummy precit
            #raise ValueError("The model has not fitted yet. Call fit() before predict()")
        

        # XGBoost retorna probabilidades, por defecto .predict() retorna la clase
        # Dependiendo de la versión, a veces .predict() retorna el label directamente;
        # si retorna probabilidades, debemos umbralizar:
        y_probs = self._model.predict(features)
        
        # Si 'y_probs' ya es [0,1], solo convertimos a int por seguridad:
        y_preds = [int(p) for p in y_probs]
        
        # O si .predict() retorna probabilidades, haríamos algo así:
        # y_probs = self._model.predict_proba(features)[:, 1]
        # y_preds = [1 if p > 0.5 else 0 for p in y_probs]

        return y_preds
    
    
