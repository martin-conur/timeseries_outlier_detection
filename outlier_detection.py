# @author:martin-conur
# Main outlier detection functions
import pandas as pd
import numpy as np

# sklearn 
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM 

from tqdm import tqdm

def detect_outliers(past_data: pd.DataFrame,
                    current_data: pd.DataFrame,
                    elasticity: int
                    )->int:
    """Classifies if 'current_data' is an outlier or not, given 'past_data'.
    
        Args:
            past_data: a sorted time indexed dataframe (ascending order) all collumns
                are considered into the outlier detection.
            current_data: a single row dataframe to evaluate if is an outlier or not, 
                'current_data' should have the same columns as 'past_data' and corresponds
                to the closest next value from 'past_data'.
            elasticity: integer from 1 to 3 that determines the level of elasticity of 
                outlier detection algorithm, being 1 the less elastic and 3 the more flexible.
        
        Returns:
            -1 for outliers, 1 for inliers
            """ 
    assert elasticity in [1, 2, 3], "elasticity should be between 1 and 3, inclusive."

    algorithms = {1: OneClassSVM,
                  2: EllipticEnvelope,
                  3: IsolationForest}
    
    # selecting the algorithmn given 'elasticity'
    outlier_detector = algorithms.get(elasticity, 3)
    # calling the outlier detector
    model = outlier_detector()
    # fitting
    model.fit(past_data)
    # calling predict method
    result = model.predict(current_data)

    return result[0]


def rolling_outlier_detector(data: pd.DataFrame,
                             window: int,
                             elasticity: int)->np.array:
    """Performs outlier detection in a rolling window fashion.
    Args:
        data: complete data, time indexed and sorted (ascending)
        window: number of past samples to consider in order to classify a new sample as outlier.
        elasticity: integer from 1 to 3 that determines the level of elasticity of 
                outlier detection algorithm, being 1 the less elastic and 3 the more flexible.
    Returns:
        A np.array with shape (data.shape[0],), -1's corresponds to outliers and 1 to inliers."""
    outlier_list = []
    for i, value in tqdm(enumerate(data.values), total=len(data)):
        if i > window:
            past_data =  data.iloc[i - window:i]
            current_data = data.iloc[i].to_frame().T
            prediction = detect_outliers(past_data, current_data, elasticity)
            outlier_list.append(prediction)
        else:
            outlier_list.append(1)
    return np.array(outlier_list)