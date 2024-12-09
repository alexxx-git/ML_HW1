from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from fastapi.responses import FileResponse
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = FastAPI()

# Load the trained model, scaler, and encoder
data= joblib.load("data.pkl")
model = data['best_ridge']
scaler = data['scaler']
encoder = data['encoder']
train_feat_order=data['order']



class CarFeatures(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: int


class CarFeaturesCollection(BaseModel):
    objects: List[CarFeatures]



@app.post("/predict_item")
def predict_item(item: CarFeatures) -> float:
    """Предсказание цены автомобиля."""

    input_df = pd.DataFrame([item.dict()])

    num_feats = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    cat_feats = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

    input_num = scaler.transform(input_df[num_feats])
    input_num_df = pd.DataFrame(input_num, columns=num_feats, index=input_df.index)

    input_cat = encoder.transform(input_df[cat_feats])
    input_cat_df = pd.DataFrame(input_cat, columns=encoder.get_feature_names_out(cat_feats), index=input_df.index)

    input_final = pd.concat([input_num_df, input_cat_df], axis=1)
    input_final=input_final[train_feat_order]
    prediction = model.predict(input_final)[0]
    return int(prediction)


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    """Прогноз цены для нескольких автомобилей"""

    df = pd.read_csv(file.file)


    num_feats = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    cat_feats = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']

    df_num = scaler.transform(df[num_feats])
    df_num_df = pd.DataFrame(df_num, columns=num_feats, index=df.index)

    df_cat = encoder.transform(df[cat_feats])
    df_cat_df = pd.DataFrame(df_cat, columns=encoder.get_feature_names_out(cat_feats), index=df.index)

    df_final = pd.concat([df_num_df, df_cat_df], axis=1)
    df_final = df_final[train_feat_order]
    predictions = model.predict(df_final)

    df['predicted_price'] = predictions

    df.to_csv("predictions.csv", index=False)

    return  FileResponse("predictions.csv", media_type="text/csv", filename="predictions.csv")