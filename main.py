from fastapi import FastAPI
from controllers.pose_controller import router as pose_router

app = FastAPI(title="Pose Detection API (YOLOv8)")

app.include_router(pose_router, prefix="/pose", tags=["Pose"])
