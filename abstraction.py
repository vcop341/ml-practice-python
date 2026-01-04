from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.ensemble import RandomForestRegressor
from oop_approach import KagglePipeline
import pandas as pd

class Main:
    reader=lambda ds:pd.read_csv(ds)
    """
    Data input:
    subm_cols-
    train_csv-__,
    y-__,
    model-
    metric-__,
    out_path-__,
    """
    def __init__(self,data):
        self.data_dict=data
        self.id=data["id"]
    def read(self,subm_cols):
        #reading dataset
        data=Main.reader(self.data_dict["train_csv"])
        
        #list with columns excluding id and target
        feats=[col for col in data.columns if col not in subm_cols];del data
        return {"X":feats,"Submission cols":subm_cols}


    def parse(self,data):
        """Data provided must contain features in a dict with X as the key"""
        #pipeline
        self.pipe=KagglePipeline(
        train_csv=self.data_dict["train_csv"],
        features=data["X"],
        target=self.data_dict["y"],
        model=self.data_dict["model"]
        )
        
        self.pipe.train()
        self.submission()
        
    def valid(self,metric):
        acc = self.pipe.validate(metric)
        print("Validation accuracy:", acc)
        pass
    def submission(self):
        self.pipe.create_submission(
        test_csv=self.data_dict["test_csv"],
        id_col=self.id,
        output_path=self.data_dict["out_path"],
        target_name=self.data_dict["y"]
        )
if __name__ == "__main__":
    dt={
        "train_csv":"C:/Users/ADMIN/Desktop/projects/kaggle subs/comp2/train.csv" ,
        "test_csv":"C:/Users/ADMIN/Desktop/projects/kaggle subs/comp2/test.csv" ,
        "out_path":"C:/Users/ADMIN/Desktop/projects/kaggle subs/comp2/submission.csv" ,
        "y":"exam_score" ,
        "id":"id" ,
        "model":RandomForestRegressor(n_estimators=1000) ,
        }
    run=Main(data=dt)
    r=run.read(subm_cols=["id","exam_score"])
    run.parse({"X":r["X"]})
    run.valid(rmse)