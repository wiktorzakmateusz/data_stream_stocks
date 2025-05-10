import pandas as pd
import numpy as np
import math
from typing import Dict, List
from river import tree
from river import metrics
from river import preprocessing
from river import feature_selection
from river import drift
from river import stats
from river import ensemble as river_ensemble
from river import base
from collections import deque, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
from sklearn.base import clone

SUPPORTED_RIVER_ENSEMBLES = ['BaggingClassifier', 'AdaBoostClassifier', 'ADWINBaggingClassifier', 'VotingClassifier', 'SRPClassifier', 'LeveragingBaggingClassifier']

class TStatFeatureSelector(base.Transformer):
    """Online t-statistics-based feature selector for binary classification."""

    _supervised = True

    def __init__(self, k: int = 10, update_interval: int = 100):
        self.k = k
        self.update_interval = update_interval
        self.feature_stats: Dict[int, Dict[str, stats.Var]] = {0: {}, 1: {}}
        self.instance_count = 0
        self.selected_features: List[str] = []

    def learn_one(self, x: dict, y: int) -> 'TStatFeatureSelector':
        self.instance_count += 1
        if y not in [0, 1]:
            raise ValueError("TStatFeatureSelector currently supports binary classification only.")

        for feature, value in x.items():
            if feature not in self.feature_stats[y]:
                self.feature_stats[y][feature] = stats.Var()
            self.feature_stats[y][feature].update(value)

        if self.instance_count % self.update_interval == 0:
            t_stats = self._compute_t_stats()
            self.selected_features = sorted(t_stats, key=t_stats.get, reverse=True)[:self.k]

        return self

    def transform_one(self, x: dict) -> dict:
        if not self.selected_features:
            return x
        return {f: x[f] for f in self.selected_features if f in x}

    def _compute_t_stats(self) -> Dict[str, float]:
        t_stats = {}
        features = set(self.feature_stats[0]) | set(self.feature_stats[1])
        for feature in features:
            s0 = self.feature_stats[0].get(feature, stats.Var())
            s1 = self.feature_stats[1].get(feature, stats.Var())
            n0, n1 = s0.n, s1.n
            if n0 > 1 and n1 > 1:
                mean0, mean1 = s0.mean.get(), s1.mean.get()
                var0, var1 = s0.get(), s1.get()
                se = math.sqrt(var0 / n0 + var1 / n1)
                if se > 0:
                    t_stats[feature] = abs((mean0 - mean1) / se)
        return t_stats

class BollingerBandDriftDetector:
    def __init__(self, window_size=20, num_std=3.5, next_drift_delay=100):
        self.window_size = window_size
        self.num_std = num_std
        self.values = deque(maxlen=window_size)
        self.drift_detected = False
        self.next_drift_delay = next_drift_delay
        self.current_next_drift_delay = 0

    def update(self, value: float):
        self.values.append(value)
        self.drift_detected = False
        self.current_next_drift_delay = max(0, self.current_next_drift_delay - 1)
        
        if len(self.values) < self.window_size:
            return
        
        mean = np.mean(self.values)
        std = np.std(self.values)
        upper_band = mean + self.num_std * std
        lower_band = mean - self.num_std * std

        if (value > upper_band or value < lower_band) and self.current_next_drift_delay == 0:
            self.drift_detected = True
            self.current_next_drift_delay = self.next_drift_delay

class FixedSizeBuffer:
    def __init__(self, max_size, num_features):
        self.max_size = max_size
        self.buffer = np.empty((0, num_features))
        self.labels = []

    def append(self, x_row):
        if len(self.buffer) >= self.max_size:
            self.buffer = np.delete(self.buffer, 0, axis=0)
            if self.labels:
                 self.labels.pop(0)
        self.buffer = np.vstack([self.buffer, x_row])

    def append_label(self, y):
        self.labels.append(y)

    def get_data(self):
        return self.buffer, np.array(self.labels)

class StockPredictor:

    def __init__(self, stock_data, drift_name, feature_selector_name, 
                 model_name=None,
                 model_names_list=None,
                 provided_model=None,
                 provided_models_list=None,
                 ensemble_strategy='majority_vote',
                 ensemble_params: dict = None,
                 learning_threshold=1000,
                 provided_detector=None):
        
        self.stock_data = stock_data
        self.data_stream = StockPredictor.ohlc_stream(stock_data)
        self.metric = metrics.ClassificationReport()
        self.learning_threshold = learning_threshold
        self.drifts_detected = 0
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_params = ensemble_params if ensemble_params is not None else {}

        self.base_models_for_reset = [] 
        self.model_names_for_reset = []
        self.original_provided_models_for_reset = []
        
        self.pipeline = None
        self.pipelines_list = []
        
        self.is_using_river_ensemble = False

        input_model_names = []
        input_provided_models = []
        is_explicit_ensemble_request = model_names_list is not None or provided_models_list is not None

        if is_explicit_ensemble_request:
            if provided_models_list is not None:
                if not isinstance(provided_models_list, list):
                    raise ValueError("provided_models_list must be a list for an ensemble.")
                if model_names_list is None or len(provided_models_list) != len(model_names_list):
                    raise ValueError("If provided_models_list is used, model_names_list must match in length.")
                input_provided_models = provided_models_list
                input_model_names = list(model_names_list)
            elif model_names_list is not None:
                if model_names_list[0].lower() not in SUPPORTED_RIVER_ENSEMBLES and not isinstance(model_names_list, list):
                    raise ValueError("model_names_list must be a list for an ensemble.")
                input_model_names = list(model_names_list)
            else:
                raise ValueError("Ensemble configuration error.")
        elif model_name is not None or provided_model is not None:
            if provided_model is not None:
                if model_name is None:
                    raise ValueError("If provided_model is used, model_name must be provided.")
                input_provided_models = [provided_model]
                input_model_names = [model_name]
            elif model_name is not None:
                input_model_names = [model_name]
            else:
                raise ValueError("Single model configuration error.")
        else:
            raise ValueError("No model configuration provided.")

        self.model_names_for_reset = input_model_names
        self.original_provided_models_for_reset = input_provided_models if input_provided_models else ([None] * len(input_model_names))

        temp_base_models = []
        if input_provided_models:
            temp_base_models = list(input_provided_models)
        else:
            for name in self.model_names_for_reset:
                model_instance, _ = StockPredictor.get_model(name)
                temp_base_models.append(model_instance)
        
        if not temp_base_models:
            raise ValueError("Base models list cannot be empty.")
        
        self.base_models_for_reset = temp_base_models

        self.is_incremental = hasattr(self.base_models_for_reset[0], 'learn_one')
        for i, model_obj in enumerate(self.base_models_for_reset):
            is_current_incremental = hasattr(model_obj, 'learn_one')
            if is_current_incremental != self.is_incremental:
                error_model_name = self.model_names_for_reset[i]
                raise ValueError(
                    f"All models must be of the same type. Model '{error_model_name}' type mismatch.")

        self.feature_selector_name = feature_selector_name
        
        known_river_ensemble_strategies = [s.lower() for s in SUPPORTED_RIVER_ENSEMBLES]

        if is_explicit_ensemble_request and self.is_incremental and self.ensemble_strategy.lower() in known_river_ensemble_strategies:
            self.is_using_river_ensemble = True
            base_models_for_river_ensemble = self.base_models_for_reset
            
            river_ensemble_model = StockPredictor.get_river_ensemble_model(
                self.ensemble_strategy, 
                base_models_for_river_ensemble,
                self.ensemble_params
            )
            fs_instance = StockPredictor.get_feature_selector(self.feature_selector_name)
            self.pipeline = StockPredictor.get_pipeline(river_ensemble_model, fs_instance)
        
        elif is_explicit_ensemble_request:
            for model_obj in self.base_models_for_reset:
                fs_instance = StockPredictor.get_feature_selector(self.feature_selector_name)
                if self.is_incremental:
                    self.pipelines_list.append(StockPredictor.get_pipeline(model_obj, fs_instance))
                else:
                    self.pipelines_list.append(StockPredictor.get_sklearn_pipeline(model_obj, fs_instance))
        else: # Single model
            fs_instance = StockPredictor.get_feature_selector(self.feature_selector_name)
            if self.is_incremental:
                self.pipeline = StockPredictor.get_pipeline(self.base_models_for_reset[0], fs_instance)
            else:
                self.pipeline = StockPredictor.get_sklearn_pipeline(self.base_models_for_reset[0], fs_instance)
        
        self.drift_name = drift_name
        self.drift_detector = provided_detector or StockPredictor.get_drift_detector(drift_name)

    @staticmethod
    def ohlc_stream(df):
        for i, row in df.iterrows():
            features = row.iloc[:-1].to_dict()
            yield i, features, row['target']

    @staticmethod
    def get_model(name: str):
        name_lower = name.lower()
        if name_lower == 'hoeffdingtreeclassifier':
            return tree.HoeffdingTreeClassifier(), True
        if name_lower == 'extremelyfastdecisiontreeclassifier':
            return tree.ExtremelyFastDecisionTreeClassifier(), True
        
        if name_lower == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=1e-4, max_iter=200, random_state=42), False
        if name_lower == 'xgboost':
            return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), False
        if name_lower == 'lgbm':
            return LGBMClassifier(verbosity=-1, random_state=42), False
        if name_lower == 'randomforest':
            return RandomForestClassifier(random_state=42), False
        else:
            raise ValueError(f"Unknown model: {name}")

    @staticmethod
    def get_drift_detector(name: str):
        name_lower = name.lower()
        if name_lower == "adwin":
            return drift.ADWIN()
        elif name_lower == "kswin":
            return drift.KSWIN(seed=42)
        elif name_lower == "dummydriftdetector":
            return drift.DummyDriftDetector()
        elif name_lower == "pagehinkley":
            return drift.PageHinkley()
        elif name_lower == 'bollingerband':
            return BollingerBandDriftDetector()
        else:
            raise ValueError(f"Unknown detector: {name}")

    @staticmethod
    def get_feature_selector(name: str):
        name_lower = name.lower()
        if name_lower == "selectkbest":
            return feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=15)
        elif name_lower == 'selectkbest_sklearn':
            return SelectKBest(score_func=f_classif, k=15)
        elif name == 'tstat':
            return TStatFeatureSelector(k=15, update_interval=100)
        else:
            raise ValueError(f"Unknown feature selector: {name}")

    @staticmethod
    def get_pipeline(model, feature_selector):
        scaler = preprocessing.StandardScaler()
        if feature_selector is None:
            return scaler | model
        return scaler | feature_selector | model

    @staticmethod
    def get_sklearn_pipeline(model, feature_selector):
        scaler = MinMaxScaler()
        steps = [('scaler', scaler)]
        if feature_selector is not None:
            steps.append(('selector', feature_selector))
        steps.append(('model', model))
        return SklearnPipeline(steps)

    @staticmethod
    def get_river_ensemble_model(strategy_name: str, base_models: list, params: dict):
        name_lower = strategy_name.lower()

        if not base_models:
            raise ValueError("base_models list cannot be empty for River ensemble.")

        if name_lower == 'baggingclassifier':
            return river_ensemble.BaggingClassifier(model=base_models[0].clone(), **params)
        elif name_lower == 'adaboostclassifier':
            return river_ensemble.AdaBoostClassifier(model=base_models[0].clone(), **params)
        elif name_lower == 'adwinbaggingclassifier':
            return river_ensemble.ADWINBaggingClassifier(model=base_models[0].clone(), **params)
        elif name_lower == 'srpclassifier':
             return river_ensemble.SRPClassifier(model=base_models[0].clone(), **params)
        elif name_lower == 'leveragingbaggingclassifier':
            return river_ensemble.LeveragingBaggingClassifier(model=base_models[0].clone(), **params)

        else:
            raise ValueError(f"Unknown or unsupported River ensemble strategy: {strategy_name}")
    
    def _combine_predictions(self, predictions: list):
        if not predictions:
            return None 
        valid_preds = [p for p in predictions if p is not None]
        if not valid_preds:
            return None

        if self.ensemble_strategy.lower() == 'majority_vote':
            vote_counts = Counter(valid_preds)
            most_common = vote_counts.most_common()
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                tied_max_vote_count = most_common[0][1]
                tied_labels = sorted([item[0] for item in most_common if item[1] == tied_max_vote_count])
                return tied_labels[0]
            return most_common[0][0]
        else:
            raise ValueError(f"Unknown manual ensembling strategy: {self.ensemble_strategy}")

    def _reset_models_and_pipelines(self):
        current_base_models = []
        for i, model_name_for_reset in enumerate(self.model_names_for_reset):
            original_prov_model = self.original_provided_models_for_reset[i]
            if original_prov_model is not None:
                if hasattr(original_prov_model, 'clone'):
                    try:
                        current_base_models.append(clone(original_prov_model))
                    except Exception:
                        model_instance, _ = StockPredictor.get_model(model_name_for_reset)
                        current_base_models.append(model_instance)
                else:
                    model_instance, _ = StockPredictor.get_model(model_name_for_reset)
                    current_base_models.append(model_instance)
            else:
                model_instance, _ = StockPredictor.get_model(model_name_for_reset)
                current_base_models.append(model_instance)
        self.base_models_for_reset = current_base_models


        if self.is_using_river_ensemble:
            river_ensemble_model = StockPredictor.get_river_ensemble_model(
                self.ensemble_strategy, 
                self.base_models_for_reset,
                self.ensemble_params
            )
            fs_instance = StockPredictor.get_feature_selector(self.feature_selector_name)
            self.pipeline = StockPredictor.get_pipeline(river_ensemble_model, fs_instance)
            self.pipelines_list = []
        elif self.pipelines_list:
            new_pipelines_list = []
            for model_obj in self.base_models_for_reset: # Iterate over new base models
                fs_instance = StockPredictor.get_feature_selector(self.feature_selector_name)
                if self.is_incremental:
                    new_pipelines_list.append(StockPredictor.get_pipeline(model_obj, fs_instance))
                else:
                    new_pipelines_list.append(StockPredictor.get_sklearn_pipeline(model_obj, fs_instance))
            self.pipelines_list = new_pipelines_list
            self.pipeline = None
        elif self.pipeline:
            fs_instance = StockPredictor.get_feature_selector(self.feature_selector_name)
            if self.is_incremental:
                self.pipeline = StockPredictor.get_pipeline(self.base_models_for_reset[0], fs_instance)
            else:
                self.pipeline = StockPredictor.get_sklearn_pipeline(self.base_models_for_reset[0], fs_instance)
            self.pipelines_list = []


    def prediction(self):
        y_final_pred = None
        if self.is_incremental:
            for stream_idx, x, y_true in self.data_stream:
                close_value = float(self.stock_data.loc[stream_idx, 'close'])
                
                if self.pipeline is not None:
                    if stream_idx >= self.learning_threshold:
                        y_final_pred = self.pipeline.predict_one(x)
                    self.pipeline.learn_one(x, y_true)
                elif self.pipelines_list:
                    y_preds_from_models = []
                    if stream_idx >= self.learning_threshold:
                        for p_line in self.pipelines_list:
                            y_preds_from_models.append(p_line.predict_one(x))
                    
                    for p_line in self.pipelines_list:
                        p_line.learn_one(x, y_true)
                    
                    if stream_idx >= self.learning_threshold:
                         y_final_pred = self._combine_predictions(y_preds_from_models)
                else:
                    raise Exception("StockPredictor not configured correctly for incremental learning.")

                if stream_idx >= self.learning_threshold and y_final_pred is not None:
                    error = int(y_final_pred != y_true)
                    if self.drift_name == 'bollingerband':
                        self.drift_detector.update(close_value)
                    else:
                        self.drift_detector.update(error)
                    self.metric.update(y_true, y_final_pred)

                    if self.drift_detector.drift_detected:
                        self.drifts_detected += 1
                        self._reset_models_and_pipelines()
        
        else:
            num_features = self.stock_data.shape[1] - 1
            buffer = FixedSizeBuffer(max_size=self.learning_threshold, num_features=num_features)
            
            for stream_idx, x_dict, y_true in self.data_stream:
                close_value = float(self.stock_data.loc[stream_idx, 'close'])
                x_array_row = np.array(list(x_dict.values()))
                
                buffer.append(x_array_row)
                buffer.append_label(y_true)
                
                should_retrain = False

                active_model_for_check = None
                if self.pipeline and hasattr(self.pipeline.steps[-1][1], 'predict'):
                    active_model_for_check = self.pipeline.steps[-1][1]
                elif self.pipelines_list and hasattr(self.pipelines_list[0].steps[-1][1], 'predict'):
                    active_model_for_check = self.pipelines_list[0].steps[-1][1]

                is_model_fitted = active_model_for_check is not None and hasattr(active_model_for_check, "classes_")


                if stream_idx >= self.learning_threshold -1 :
                    if not is_model_fitted or (stream_idx + 1) % self.learning_threshold == 0:
                        should_retrain = True
                    if hasattr(self.drift_detector, 'drift_detected') and self.drift_detector.drift_detected:
                        should_retrain = True

                if should_retrain and len(buffer.labels) >= self.learning_threshold * 0.1 and len(buffer.labels) > 1: # Ensure some data in buffer
                    X_train, y_train = buffer.get_data()
                    if X_train.shape[0] > 0 and len(np.unique(y_train)) > 1 : # Ensure multiple classes for some classifiers
                        pipelines_to_train = self.pipelines_list if self.pipelines_list else ([self.pipeline] if self.pipeline else [])
                        for p_line in pipelines_to_train:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                p_line.fit(X_train, y_train)
                        is_model_fitted = True

                y_final_pred = None
                if stream_idx >= self.learning_threshold and is_model_fitted:
                    x_array_for_predict = x_array_row.reshape(1, -1)
                    if self.pipeline is not None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                y_final_pred = self.pipeline.predict(x_array_for_predict)[0]
                            except Exception: y_final_pred = None
                    elif self.pipelines_list:
                        y_preds_from_models = []
                        for p_line in self.pipelines_list:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                try:
                                    y_preds_from_models.append(p_line.predict(x_array_for_predict)[0])
                                except Exception: y_preds_from_models.append(None)
                        y_final_pred = self._combine_predictions(y_preds_from_models)
                
                if y_final_pred is not None:
                    error = int(y_final_pred != y_true)
                    if self.drift_name == 'bollingerband':
                        self.drift_detector.update(close_value)
                    else:
                        self.drift_detector.update(error)
                    self.metric.update(y_true, y_final_pred)

                    if self.drift_detector.drift_detected:
                        self.drifts_detected += 1
                        self._reset_models_and_pipelines()


        accuracy, metrics_result_df = self.get_metrics()
        return accuracy, metrics_result_df

    def get_metrics(self):
        print(self.metric)
        cm_classes = self.metric.cm.classes if self.metric.cm is not None else []
        self.metric._precisions = {}
        self.metric._recalls = {}
        for c_val in cm_classes:
            if c_val not in self.metric._f1s:
                self.metric._f1s[c_val] = metrics.F1(cm=self.metric.cm, pos_val=c_val)
            if c_val not in self.metric._precisions:
                 self.metric._precisions[c_val] = metrics.Precision(cm=self.metric.cm, pos_val=c_val)
            if c_val not in self.metric._recalls:
                 self.metric._recalls[c_val] = metrics.Recall(cm=self.metric.cm, pos_val=c_val)


        accuracy = self.metric._accuracy
        
        report_data = []
        sorted_classes = sorted(cm_classes)

        for class_val in sorted_classes:
            precision = self.metric._precisions.get(class_val).get() if self.metric._precisions.get(class_val) else 0.0
            recall = self.metric._recalls.get(class_val).get() if self.metric._recalls.get(class_val) else 0.0
            f1 = self.metric._f1s.get(class_val).get() if self.metric._f1s.get(class_val) else 0.0
            report_data.append([class_val, precision, recall, f1])
        
        if not report_data and (0 in cm_classes or 1 in cm_classes): 
            if 0 not in sorted_classes and 0 in cm_classes: sorted_classes.append(0)
            if 1 not in sorted_classes and 1 in cm_classes: sorted_classes.append(1)
            sorted_classes = sorted(list(set(sorted_classes)))
            for class_val in sorted_classes :
                 report_data.append([class_val, 0.0, 0.0, 0.0])


        metrics_result_df = pd.DataFrame(report_data, columns=['class', 'precision', 'recall', 'f1'])
        metrics_result_df = metrics_result_df.round(3)
        
        return round(accuracy.get() if accuracy is not None else 0.0, 3), metrics_result_df
