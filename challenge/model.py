import pandas as pd
import numpy as np

from typing import Tuple, Union, List

# Models and tools
import xgboost as xgb
from datetime import datetime

###############################################################################
#                          DEFINITION OF CLASSES                              #
###############################################################################


class DelayModel:
    
    def __init__(self):
        """
        Initialize the DelayModel with a placeholder for the trained model
        and any other essential attributes.
        """
        self._model = None  
        
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
        
        # BALANCE FACTOR, JUST IN CASE
        
        self._scale_pos_weight = 1.0

    def _get_period_day(self, date_str: str) -> str:
        """
        Determine the period of the day (morning, afternoon, night) 
        based on the flight date/time string.

        Args:
            date_str (str): Date string with format '%Y-%m-%d %H:%M:%S'.

        Returns:
            str: 'morning', 'afternoon', or 'night'.
        """
        date_time = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').time()

        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        
        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        else:
            return 'noche'

    def _is_high_season(self, date_str: str) -> int:
        """
        Determine if a given date/time string falls into the "high season" period.

        Args:
            date_str (str): A date/time string in '%Y-%m-%d %H:%M:%S' format.

        Returns:
            int: 1 if the date is in high season, 0 otherwise.
        """
        fecha = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        año = fecha.year


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
        Calculate the difference in minutes between 'Fecha-O' and 'Fecha-I' for a given row.

        Args:
            row (pd.Series): A row that must include 'Fecha-O' and 'Fecha-I'.

        Returns:
            float: The difference in minutes (Fecha-O - Fecha-I).
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
        Prepare raw flight data for model training or prediction.

        Steps:
        1) Create derived columns such as period of day, high season, 
           min_diff (and 'delay' if thresholds apply).
        2) Convert categorical columns to dummies.
        3) Return features (and target if 'target_column' is specified).

        Args:
            data (pd.DataFrame): The raw flight data.
            target_column (str, optional): If set, this column is returned as y.

        Returns:
            (pd.DataFrame, pd.DataFrame): if target_column is provided.
            pd.DataFrame: if target_column is not provided.
        """

        df = data.copy()

        
        if 'Fecha-I' in df.columns:
            df['period_day'] = df['Fecha-I'].apply(self._get_period_day)
            df['high_season'] = df['Fecha-I'].apply(self._is_high_season)

        if 'Fecha-O' in df.columns and 'Fecha-I' in df.columns:
            df['min_diff'] = df.apply(self._get_min_diff, axis=1)
            
            df['delay'] = np.where(df['min_diff'] > 15, 1, 0)

        # Convert categorical columns to dummies    
        feature_df = []

        if 'OPERA' in df.columns:
            feature_df.append(pd.get_dummies(df['OPERA'], prefix='OPERA'))
        if 'TIPOVUELO' in df.columns:
            feature_df.append(pd.get_dummies(df['TIPOVUELO'], prefix='TIPOVUELO'))
        if 'MES' in df.columns:
            feature_df.append(pd.get_dummies(df['MES'], prefix='MES'))

        
        if feature_df:
            X = pd.concat(feature_df, axis=1)
        else:
            X = pd.DataFrame()

        X = X.reindex(columns=self._top_10_features, fill_value=0)

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

        target_series = target.squeeze()

        # Ensure target_series is a Series
        if isinstance(target_series, pd.DataFrame):
            target_series = target_series.iloc[:, 0]

        # Calculate the scale_pos_weight to balance the classes
        n_delay_0 = len(target_series[target_series == 0])
        n_delay_1 = len(target_series[target_series == 1])
        self._scale_pos_weight = n_delay_0 / n_delay_1 if n_delay_1 else 1.0

       
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
        
        if self._model is None:
            return [0] * len(features)  #dummy precit
            #raise ValueError("The model has not fitted yet. Call fit() before predict()")
        
        y_probs = self._model.predict(features)
        
        y_preds = [int(p) for p in y_probs]
        
        return y_preds
    