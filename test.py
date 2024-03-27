import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_image_sim(image_path, text_list):
    global model
    global processor
    image = Image.open(image_path).convert('RGB')
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)
    print(type(inputs))
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    print(logits_per_image.size())
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

if __name__ == "__main__":
    image_path = 'cards/card1.jpg'
    tlist = ['violin', 'women', 'musician']
    print(text_image_sim(image_path, tlist))
