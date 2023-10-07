from src.image_search import ImageSearch


def inference(path):
    img = ImageSearch()
    embed = img.embed()
    image = img.search_image(embed, path)
    path_image = img.search_path_image(embed, path)

    return image, path_image