from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pickle

# processor 
# Image -> pixel values

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):

     # resize image (IMPORTANT)
    image = image.resize((224, 224))
    inputs = processor(image, return_tensors="pt") # rerturn_tensors="pt" means return PyTorch tensors
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True) # output[0] to get the first (and only) generated caption
    return caption




