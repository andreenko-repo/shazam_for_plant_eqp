import os
from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "industrial_equipment"
DATA_PATH = "./data"


print("Loading embedding model...")
model = SentenceTransformer("clip-ViT-B-32")
VECTOR_SIZE = 512

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_qdrant_collection():
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
        print("Collection created successfully.")


def process_and_upload_data():
    equipment_folders = [
        d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))
    ]

    for equipment_name in equipment_folders:
        folder_path = os.path.join(DATA_PATH, equipment_name)

        info_file_path = os.path.join(folder_path, "info.txt")
        equipment_info = ""
        if os.path.exists(info_file_path):
            with open(info_file_path, "r", encoding="utf-8") as f:
                equipment_info = f.read()
            print(f"\nProcessing: {equipment_name}")
        else:
            print(f"\nSkipping {equipment_name}: 'info.txt' not found.")
            continue

        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            print(f"No images found for {equipment_name}.")
            continue

        points_to_upload = []

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            try:
                image = Image.open(image_path)

                embedding = model.encode(image, convert_to_tensor=True)

                payload = {
                    "equipment_name": equipment_name,
                    "information": equipment_info,
                    "image_source": image_path,
                }

                point = models.PointStruct(
                    id=str(uuid.uuid4()), vector=embedding.tolist(), payload=payload
                )
                points_to_upload.append(point)

                print(f"  - Processed image: {image_file}")

            except Exception as e:
                print(f"  - Error processing image {image_file}: {e}")

        if points_to_upload:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upload,
                wait=True,
            )
            print(
                f"Uploaded {len(points_to_upload)} images for {equipment_name} to Qdrant."
            )


if __name__ == "__main__":
    create_qdrant_collection()
    process_and_upload_data()

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"\nTotal points in collection: {collection_info.points_count}")
