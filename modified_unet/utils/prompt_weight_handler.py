from models.base_unet import BaseUNet
from models.detailed_unet import DetailedUNet
from models.semantic_unet import SemanticUNet

def map_prompt_to_model(prompt_weight):
    if prompt_weight > 0.8:
        return DetailedUNet()
    elif 0.5 < prompt_weight <= 0.8:
        return SemanticUNet()
    else:
        return BaseUNet()

def get_word_weight(word):
    word_weights = {
        'detail': 0.9,
        'fine': 0.85,
        'semantic': 0.7,
        'general': 0.4
    }
    return word_weights.get(word, 0.5) 

def process_prompt(prompt):
    words = prompt.split() 
    selected_models = []

    for word in words:
        word_weight = get_word_weight(word)
        print(f"Word: '{word}' - Weight: {word_weight}")
        model = map_prompt_to_model(word_weight)
        selected_models.append((word, model))

    return selected_models
