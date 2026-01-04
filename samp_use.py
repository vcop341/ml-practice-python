from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from kaggle_pipeline import KagglePipeline
features = [
    "HomePlanet", "CryoSleep", "Cabin", "Destination",
    "Age", "VIP", "RoomService", "FoodCourt",
    "ShoppingMall", "Spa", "VRDeck"
]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

pipeline = KagglePipeline(
    train_csv="train.csv",
    features=features,
    target="Transported",
    model=model
)

pipeline.train()

acc = pipeline.validate(accuracy_score)
print("Validation accuracy:", acc)

pipeline.create_submission(
    test_csv="test.csv",
    id_col="PassengerId",
    output_path="submission.csv",
    target_name="Transported"
)
