import io

from PIL import Image

MAX_SIZE = 1920
QUALITY = 90


def normalize_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) > MAX_SIZE:
        image.thumbnail((MAX_SIZE, MAX_SIZE))
    return image


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=QUALITY)
    return buffer.getvalue()
