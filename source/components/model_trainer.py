import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define models
            models = {
                "linear_regression": LinearRegression(),
                "ridge": Ridge(),
                "lasso": Lasso(),
                "decision_tree": DecisionTreeRegressor(),
                "random_forest": RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
                "adaboost": AdaBoostRegressor(),
                "xgboost": XGBRegressor(),
                "catboost": CatBoostRegressor(verbose=False),
                "k_nearest": KNeighborsRegressor()
            }

            # Evaluate models
            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models
            )

            # Select best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with r2_score > 0.6")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predictions and final score
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)