import cv2
import csv
from glob import glob
from pathlib import Path
from statistics import mean

from towhee.types.image import Image
from towhee import pipe, ops, DataCollection
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


class ImageSearch:

    def __init__(self):
        self.model = "resnet50"
        self.device = None
        self. host = "127.0.0.1"
        self.port = "19530"
        self.topk = 10
        self.dim = 2048
        self.collection_name = "image_search"
        self.index_type = "IVF_FLAT"
        self.metric_type = "L2"
        self.connection = connections.connect(host=self.host, port=self.port)
    

    def load_image(self, path):
        if path.endswith('csv'):
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                for item in reader:
                    yield item[1]
        else:
            for item in glob(path):
                yield item


    def create_milvus_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        
        fields = [FieldSchema(name='path',
                              dtype=DataType.VARCHAR,
                              description='path to image',
                              max_length=500,
                              is_primary=True,
                              auto_id=False),

                  FieldSchema(name='embedding',
                              dtype=DataType.FLOAT_VECTOR,
                              description='image embedding vectors',
                              dim=self.dim)
                ]
        
        schema = CollectionSchema(fields=fields, 
                                  description='reverse image search')
        collection = Collection(name=self.collection_name,
                                schema=schema)

        index_params = {
            'metric_type': self.metric_type,
            'index_type': self.index_type,
            'params': {"nlist": 2048}
        }
        collection.create_index(field_name='embedding', index_params=index_params)

        return collection


    def embed(self):
        p_embed = (pipe.input('src')
                   .flat_map('src', 'img_path', self.load_image)
                   .map('img_path', 'img', ops.image_decode())
                   .map('img', 'vec', ops.image_embedding.timm(model_name=self.model,
                                                               device=self.device)))

        return p_embed
    

    def insert(self, p_embed):
        collection = self.create_milvus_collection()
        p_insert = (p_embed.map(('img_path', 'vec'), 'mr', ops.ann_insert.milvus_client(
                    host=self.host,
                    port=self.port,
                    collection_name=self.collection_name))
                    .output('mr'))
        
        return collection, p_insert


    def read_images(self, img_paths):
        imgs = []
        for p in img_paths:
            imgs.append(Image(cv2.imread(p), 'BGR'))

        return imgs

    
    def path_image(self, img_paths):
        imgs_path = []
        for p in img_paths:
            imgs_path.append(p)

        return img_paths
    

    def search_image(self, p_embed, path):
        p_search_pre = (p_embed.map('vec', ('search_res'), ops.ann_search.milvus_client(
                            host=self.host, port=self.port, limit=self.topk,
                            collection_name=self.collection_name))
                    .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x]))
        p_search_img = (p_search_pre.map('pred', 'pred_images', self.read_images)
                        .output('img', 'pred_images'))
        
        return DataCollection(p_search_img(path)).show()
    

    def search_path_image(self, p_embed, path):
        p_search_pre = (p_embed.map('vec', ('search_res'), ops.ann_search.milvus_client(
                            host=self.host, port=self.port, limit=self.topk,
                            collection_name=self.collection_name))
                    .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x]))
        p_search_path = (p_search_pre.map('pred','pred_images', self.path_image)
                    .output('img_path', 'pred_images'))
        
        return DataCollection(p_search_path(path)).to_list()
    


    

