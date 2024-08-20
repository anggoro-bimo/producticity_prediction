import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='/home/er_bim/productivity-prediction/artifacts/model.pkl'
            preprocessor_path='/home/er_bim/productivity-prediction/artifacts/preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        day: str,
        week: str, 
        department: str, 
        team: int, 
        targeted_productivity: float, 
        smv: float, 
        wip: int, 
        over_time: int, 
        incentive: int,
        no_of_style_change: int, 
        no_of_workers: int):

        self.day = day
        
        self.week = week

        self.department = department

        self.team = team

        self.targeted_productivity = targeted_productivity

        self.smv = smv

        self.wip = wip
        
        self.over_time = over_time
        
        self.incentive = incentive
        
        self.no_of_style_change = no_of_style_change
        
        self.no_of_workers = no_of_workers

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "day": [self.day],
                "week": [self.week],
                "department": [self.department],
                "team": [self.team],
                "targeted_productivity": [self.targeted_productivity],
                "smv": [self.smv],
                "wip": [self.wip],
                "over_time": [self.over_time],
                "incentive": [self.incentive],
                "no_of_style_change": [self.no_of_style_change],
                "no_of_workers": [self.no_of_workers]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)