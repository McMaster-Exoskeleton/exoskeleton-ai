from fastapi import FastAPI

class InferenceRequest(BaseModel):
    


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def read_health():
    return {"status": "ok"}

@app.get("/predict")


# mass of the partipant matters
# scale the output of the model by the mass of the participant
# 