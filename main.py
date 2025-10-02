from fastapi import FastAPI
from controllers.pose_controller import router as pose_router

app = FastAPI(title="Pose Detection API", version="1.0")

# Include controllers
app.include_router(pose_router, prefix="/pose", tags=["Pose Detection"])

@app.get("/")
def root():
    return {"status": "Backend running 🚀"}
