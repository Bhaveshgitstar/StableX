import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_models import UNet, ModifiedUNet
from data_loader import get_data_loader

def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "prompt_weight" in line:
                weight = float(line.split(":")[-1].strip())
                return weight
    raise ValueError("Log file does not contain a valid 'prompt_weight' entry.")

def select_unet_from_log(log_file_path):
    prompt_weight = parse_log_file(log_file_path)
    if prompt_weight >= 0.7:
        print(f"Using ModifiedUNet (Prompt Weight: {prompt_weight})")
        return ModifiedUNet(), prompt_weight
    else:
        print(f"Using UNet (Prompt Weight: {prompt_weight})")
        return UNet(), prompt_weight
      
def select_model(prompt_weight):
    if prompt_weight < 0.5:
        print("Selected: Basic UNet")
        return UNet()
    else:
        print("Selected: Modified UNet")
        return ModifiedUNet()

def apply_pipeline(input_tensor, progression_stages):
    current_output = input_tensor
    for stage in progression_stages:
        prompt_weight = stage["prompt_weight"]
        model = select_model(prompt_weight)
        current_output = model(current_output)
        print(f"Stage '{stage['stage']}': Output shape {current_output.shape}")
    return current_output


def train_unet(model, dataloader, epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    log_file_path = "path_to_log_file.log"  
    selected_unet, prompt_weight = select_unet_from_log(log_file_path)

    data_dir = "path_to_your_dataset"
    dataloader = get_data_loader(data_dir, batch_size=8)

    train_unet(selected_unet, dataloader)
