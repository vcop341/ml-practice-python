import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataProcessor:
    """
    Handles preprocessing only (SRP)
    """
    def __init__(self):
        self.pipeline = None

    def build(self, X: pd.DataFrame):
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = X.select_dtypes(include=["object", "bool"]).columns

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.pipeline = ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ])

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        return self.pipeline.transform(X)


class ModelTrainer:
    """
    Model-agnostic training (Dependency Injection)
    """
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class KagglePipeline:
    """
    Orchestrates everything (Facade pattern)
    """
    def __init__(
        self,
        train_csv: str,
        features: list,
        target: str,
        model,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.train_csv = train_csv
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.processor = DataProcessor()
        self.trainer = ModelTrainer(model)

        self._load_data()
        self._prepare_data()

    def _load_data(self):
        self.df = pd.read_csv(self.train_csv)
        self.X = self.df[self.features]
        self.y = self.df[self.target]

    def _prepare_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.processor.build(self.X_train)
        self.X_train = self.processor.fit_transform(self.X_train)
        self.X_val = self.processor.transform(self.X_val)

    def train(self):
        self.trainer.train(self.X_train, self.y_train)

    def validate(self, metric_fn):
        preds = self.trainer.predict(self.X_val)
        return metric_fn(self.y_val, preds)

    def predict_test(self, test_csv: str):
        test_df = pd.read_csv(test_csv)
        X_test = self.processor.transform(test_df[self.features])
        return self.trainer.predict(X_test)

    def create_submission(
        self,
        test_csv: str,
        id_col: str,
        output_path: str,
        target_name: str
    ):
        test_df = pd.read_csv(test_csv)
        preds = self.predict_test(test_csv)

        submission = pd.DataFrame({
            id_col: test_df[id_col],
            target_name: preds
        })

        submission.to_csv(output_path, index=False)
        print("✅ Submission file created:", output_path)
