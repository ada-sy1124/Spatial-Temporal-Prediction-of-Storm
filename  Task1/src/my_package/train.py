# src/yourpkg/train.py
import torch
from tqdm.auto import tqdm
from livelossplot import PlotLosses


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    nepochs,
    input_len,
    total_len,
    save_path
):
    """
    Executes a multi-epoch training and validation loop with live loss plotting.
    
    Args:
        model: The PyTorch model to train.
        optimizer, scheduler, criterion: Optimization components.
        device: 'cuda' or 'cpu'.
        input_len, total_len: Sequence length parameters for the model forward pass.
        save_path: Location to save the model weights with the lowest validation loss.
    """

    liveloss = PlotLosses()
    best_val_loss = float("inf")

    def train_epoch(model, loader):
        """Standard backpropagation pass for one epoch."""
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc="Train", leave=False)

        for x_in, x_out in pbar:
            x_in, x_out = x_in.to(device), x_out.to(device)

            optimizer.zero_grad()

            # Forward pass using sequence length constraints
            pred = model(x_in, input_len=input_len, total_len=total_len)
            loss = criterion(pred, x_out)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return running_loss / len(loader)

    @torch.no_grad()
    def valid_epoch(model, loader):
        """Evaluation pass on the validation set."""
        model.eval()
        running_loss = 0.0
        pbar = tqdm(loader, desc="Valid", leave=False)

        for x_in, x_out in pbar:
            x_in, x_out = x_in.to(device), x_out.to(device)
            pred = model(x_in, input_len=input_len, total_len=total_len)
            loss = criterion(pred, x_out)
            running_loss += loss.item()

        return running_loss / len(loader)

    
    # Main training loop
    for epoch in range(1, nepochs + 1):
        logs = {}
        print(f"Epoch {epoch}/{nepochs}")

        train_loss = train_epoch(model, train_loader)
        val_loss   = valid_epoch(model, val_loader)

        # Dictionary keys matching livelossplot expectations
        # logs['' + 'MAE loss'] = train_loss
        # logs['val_' +'MAE loss'] = val_loss
        liveloss.update({
            'mae': train_loss,      # 训练集叫 'mae'
            'val_mae': val_loss     # 验证集叫 'val_mae'
        })

        # Update and render the live training progress chart
        # liveloss.update(logs)
        liveloss.draw()

        # Checkpointing logic based on validation performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model! Loss: {best_val_loss:.6f}")

        print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        print("-" * 50)
