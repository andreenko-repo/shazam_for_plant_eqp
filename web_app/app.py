import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "industrial_equipment"

app = Flask(__name__)

# --- Load Models and Clients ---
try:
    print("Loading embedding model...")
    model = SentenceTransformer("clip-ViT-B-32")
    print("Model loaded successfully.")

    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print("Successfully connected to Qdrant and collection.")

except Exception as e:
    print(f"Error during initialization: {e}")

    model = None
    qdrant_client = None


# --- Web Routes ---
@app.route("/")
def index():
    """Render the main page with the camera view."""
    return render_template("index.html")


@app.route("/identify", methods=["POST"])
def identify():
    """
    API endpoint to receive an image, find the best match, and return info.
    """
    if not model or not qdrant_client:
        return jsonify({"error": "Server is not ready. Models or DB not loaded."}), 500

    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    header, encoded = data["image"].split(",", 1)
    image_data = base64.b64decode(encoded)

    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    # --- Perform the Search ---
    query_embedding = model.encode(image, convert_to_tensor=True)

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=1,
    )

    results = []
    if search_results:
        for result in search_results:
            results.append({"score": result.score, "payload": result.payload})

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
