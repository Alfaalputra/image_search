import os, sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

path_this = os.path.abspath(os.path.dirname(__file__))
path_root = os.path.join(path_this, "..")
sys.path.append(path_root)

from src.image_search import ImageSearch

app = FastAPI()

class Image(BaseModel):
    path: str = Field("/home/alfa/jupyter/image/test.jpg",
                      description="path to the image as input")
    

class Dataset(BaseModel):
    path: str = Field("data/reverse_image_search.csv",
                      description="path to the dataset as input in csv file")
    

@app.post("/upload_dataset")
async def upload_dataset(dataset: Dataset):
    path = dataset.path
    img = ImageSearch()
    # collection = img.create_milvus_collection()
    embed = img.embed()
    collection, insert = img.insert(p_embed=embed)
    insert(path)

    return {"numbered of insert data": collection.num_entities}



@app.post("/search_image")
async def search_image(image: Image):
    path = image.path
    img = ImageSearch()
    embed = img.embed()
    path_image = img.search_path_image(embed, path)

    return path_image


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6900)