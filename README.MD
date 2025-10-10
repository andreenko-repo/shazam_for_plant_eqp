# Shazam for Plant Equipment

This project is a simple image recognition application designed to identify industrial plant equipment using a mobile phone's camera. Think of it as a "Shazam for industrial equipment."

The application does not require internet connection and all the data is strored locally.

You point your phone's camera at a piece of equipment, capture an image, and the application identifies it, providing information and documentation links (or any other information you put in the databse).

It uses the CLIP-ViT-B-32 model to generate vector embeddings from images and stores them in a local Qdrant vector database for fast and efficient similarity searching.

## How It Works

1.  **Data Ingestion**: The `ingest_data.py` script processes a local dataset of equipment images. For each image, it generates a vector embedding using the CLIP model.
2.  **Vector Database**: These embeddings, along with metadata (equipment name, manufacturer, documentation), are stored in a Qdrant vector database collection.
3.  **Web Interface**: A simple web app allows users to access their camera.
4.  **Identification**: The user captures an image. The frontend sends this image to the backend, which generates a new vector embedding.
5.  **Similarity Search**: This new embedding is used to query the Qdrant database, which finds the most similar image vectors from its collection.
6.  **Display Results**: The metadata of the best match is returned and displayed to the user.

## Tech Stack

*   **Backend**: Python, Flask
*   **ML/Vector Search**:
    *   `sentence-transformers` for the CLIP model
    *   Qdrant as the vector database
*   **Frontend**: HTML, CSS, JavaScript
*   **Deployment**: Gunicorn, Docker (for Qdrant)

## Project Structure

```
shazam_for_plant_eqp/
├── data/                 # Dataset of equipment images and info
│   ├── equipment_1/
│   │   ├── image1.jpg
│   │   └── info.txt
│   └── ...
├── web_app/              # Flask web application
│   ├── static/
│   ├── templates/
│   └── app.py
├── docker-compose.yml    # Docker configuration for Qdrant
├── ingest_data.py        # Script to populate the vector database
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/shazam_for_plant_eqp.git
cd shazam_for_plant_eqp
```

### 2. Prepare the Dataset
Create a `data` directory. Inside, create a sub-folder for each piece of equipment. Each sub-folder should contain images of the equipment and an `info.txt` file with its details.

```
data/
│   ├── equipment_1/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── info.txt
│   └── equipment_2/
│       ├── image1.jpg
│       └── info.txt
```

**Example `info.txt`:**
```
Manufacturer: Yokogawa Electric
Name: EJX110A Differential Pressure Transmitter
Documentation: https://web-material3.yokogawa.com/GS01C25B01-01EN.pdf
Notes: Yokogawa DP transmitter
```

### 3. Start Qdrant Vector Database
You need Docker and Docker Compose installed. The `docker-compose.yml` file in the repository will configure and start a Qdrant container. Data will be persisted in the `qdrant_storage` directory.

```bash
docker-compose up -d
```

### 4. Set Up Python Environment
Create and activate a virtual environment.

```bash
# Create virtual environment
python -m venv py_shazam_plant_eqp

# Activate it (macOS/Linux)
source py_shazam_plant_eqp/bin/activate
# Or on Windows
# .\py_shazam_plant_eqp\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 5. Ingest Data into Qdrant
Run the ingestion script to process your dataset and populate the Qdrant database. This will create a collection named `industrial_equipment`.

```bash
python ingest_data.py
```

### 6. Generate SSL Certificate (for Mobile Testing)
To use your phone's camera, the web app must be served over HTTPS. You can generate a self-signed certificate for local testing.

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 14 -nodes
```
This will create `key.pem` and `cert.pem` files.

### 7. Run the Web Application
Start the Flask application using Gunicorn. The app will be accessible on all network interfaces on port 5001.

```bash
cd web_app
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5001 --certfile=../cert.pem --keyfile=../key.pem app:web_app
```

### 8. Access the App
Open a web browser on your mobile device (connected to the same network) and navigate to:
`https://<your_computer_ip>:5001`

Grant camera permissions when prompted, and you're ready to start identifying equipment!
