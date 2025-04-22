from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image

app = FastAPI()

# CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "X-ray Disease Predictor is Live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction = await predict_image(file)
    return {"prediction": prediction}
