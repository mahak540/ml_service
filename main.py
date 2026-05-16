from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionPipeline
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi.responses import FileResponse
import os
import uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# STATIC FILES

app.mount(
    "/generated",
    StaticFiles(directory="generated"),
    name="generated"
)

# CREATE FOLDER

os.makedirs("generated", exist_ok=True)

# LOAD MODEL

pipe = StableDiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

# REQUEST MODEL

class PromptRequest(BaseModel):
    prompt:str

# ROUTE

@app.post("/generate")
async def generate_image(data:PromptRequest):

    prompt = data.prompt

    image = pipe(prompt).images[0]
    filename = f"{uuid.uuid4()}.png"
    image_path = f"generated/{filename}"

    image.save(image_path)

    return {
        "image": f"http://127.0.0.1:8000/generated/{filename}"
    }
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"generated/{filename}"
    return FileResponse(file_path, filename=filename)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/video")
def generate_video(data: PromptRequest):

    filename = f"{uuid.uuid4()}.mp4"
    path = f"generated/{filename}"

    # TEMP MOCK (important for now)
    with open(path, "wb") as f:
        f.write(b"fake-video-content")

    return {
        "video": f"http://127.0.0.1:8000/generated/{filename}"
    }