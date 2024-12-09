This directory will contain all experimental results, regardless of model variant, organized by some standard format.

import torch
import keyboard

# Define a flag to stop training
stop_training = False

# Define a function to handle keypress
def on_key_press():
    global stop_training
    if keyboard.is_pressed('q'):  # Listen for 'q' key press
        print("Kill switch activated. Stopping training...")
        stop_training = True

# Example training loop
def train_model(model, dataloader, optimizer, criterion, num_epochs):
    global stop_training
    for epoch in range(num_epochs):
        if stop_training:
            break
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if stop_training:
                break
            # Forward pass
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check if the kill switch is activated
        on_key_press()

    # Cleanup
    print("Releasing GPU memory...")
    torch.cuda.empty_cache()
    del model, optimizer
    print("Training stopped safely.")

# Dummy example
if __name__ == "__main__":
    # Create dummy model, dataloader, optimizer, and criterion
    model = torch.nn.Linear(10, 2).cuda()
    dataloader = [(
        torch.randn(16, 10).cuda(),
        torch.randint(0, 2, (16,)).cuda()
    ) for _ in range(100)]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("Press 'q' to stop training safely.")
    train_model(model, dataloader, optimizer, criterion, num_epochs=10)
