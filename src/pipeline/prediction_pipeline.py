import os, sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        preprocessor_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
        model_path = os.path.join("artifacts/model_trainer", "model.pkl")

        processor = load_object(preprocessor_path)
        model = load_object(model_path)

        scaled = processor.transform(features)
        pred = model.predict(scaled)

        return pred
    


class CustomClass:
    def __init__(self,
                 age:int,
                 workclass:int,
                 education-num:int,
                 marital-status:int,
                 occupation:int,
                 relationship:int, 
                 race:int,
                 gender:int,
                 capital-gain:int,
                 capital-loss:int,
                 hours-per-week:int,
                 native-country:int):
        self.age = age,
        self.workclass = workclass,
        self.education-num = education-num,
        self.marital-status = marital-status,
        self.occupation = occupation,
        self.relationship = relationship,
        self.race = race,
        self.gender = gender,
        self.capital-gain = capital-gain,
        self.capital-loss = capital-loss,
        self.hours-per-week = hours-per-week,
        self.native-country = native-country,

    
    def get_data_DataFrame(self):
        try:
            custom_input = {
                "age": [self.age],
                "workclass": [self.workclass],
                "education-num": [self.education-num],
                "marital-status": [self.marital-status],
                "occupation": [self.occupation],
                "relationship": [self.relationship],
                "race": [self.race],
                "gender": [self.gender],
                "capital-gain": [self.capital-gain],
                "capital-loss": [self.capital-loss],
                "hours-per-week": [self.hours-per-week],
                "native-country": [self.native-country]
            }

            data = pd.DataFrame(custom_input)

            return data
        
        except Exception as e:
            raise CustomException(e, sys)


