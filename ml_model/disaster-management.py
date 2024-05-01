# Import necessary libraries
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.webservice import Webservice
import rasterio
import numpy as np
import requests
import json

# Set up the workspace
ws = Workspace.from_config()

# Connect to the registered model in Azure ML Studio
model = Model(ws, "your_model_name")

# Connect to the web service that's hosting the model
service = Webservice(ws, "your_service_name")
service_url = service.scoring_uri
service_key = service.get_keys()[0] if len(service.get_keys()) > 0 else None


# Read a TIF file using rasterio
def read_tif(tif_path):
    with rasterio.open(tif_path) as src:
        image = (
            src.read()
        )  # Reading the TIF file as an array# Perform any additional preprocessing needed here
        return image


# Prepare the image for prediction
def prepare_image(image_array):
    # Convert the image to a format your model expects# For example, converting to JSON
    image_json = json.dumps({"data": image_array.tolist()})
    return image_json


# Send the image to the model for categorization
def categorize_image(image_json, service_url, service_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {service_key}",
    }  # Make the request and display the response
    response = requests.post(service_url, headers=headers, data=image_json)
    return response.json()


# The path to your TIF file
tif_path = "/path/to/your/image.tif"  # Process the image and get predictions
image_array = read_tif(tif_path)
image_json = prepare_image(image_array)
result = categorize_image(image_json, service_url, service_key)

# Display the categorization result
print("Categorization result:", result)
