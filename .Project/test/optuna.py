import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Utils.train import train_eval
from Utils.models import CNN, UNet, ConvTasNet  
from Utils.dataset import DynamicBuckets, StaticBuckets, PTODataset, pto_collate, BucketSampler


import config


# === Define Optuna Objective Function ===
def objective(trial):

    # === Hyperparameter Search Space ===
    enc_dim = trial.suggest_categorical("enc_dim", [64, 128, 256]) 
    feature_dim = trial.suggest_categorical("feature_dim", [32, 48, 64]) 
    kernel_size = trial.suggest_categorical("kernel_size", [(3, 3), (5, 5)]) 
    num_layers = trial.suggest_int("num_layers", 2, 6)  
    num_stacks = trial.suggest_int("num_stacks", 1, 3) 
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)  
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4]) 

    # === Loading Configurations ===
    dataset_dir = config.DATASET_DIR
    sr = config.SAMPLE_RATE
    n_fft = config.N_FFT
    hop_length = config.HOP_LENGTH
    model_name = config.MODEL
    pad_method = config.PAD_METHOD
    num_bucket = config.NUM_BUCKET
    bucket_sizes = [sr, sr * 2, sr * 3, sr * 4, sr * 5]

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Dataset ===
    print(f"Using `{pad_method}` dataset padding method.")
    if pad_method == "dynamic":
        train_dataset = DynamicBuckets(dataset_dir, "trainset_56spk", sr, n_fft, hop_length, num_bucket)
        val_dataset = DynamicBuckets(dataset_dir, "trainset_28spk", sr, n_fft, hop_length, num_bucket)

        train_sampler = BucketSampler(train_dataset.bucket_indices, batch_size=batch_size)
        val_sampler = BucketSampler(val_dataset.bucket_indices, batch_size=batch_size)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)

    elif pad_method == "static":
        train_dataset = StaticBuckets(dataset_dir, "trainset_56spk", sr, n_fft, hop_length, bucket_sizes)
        val_dataset = StaticBuckets(dataset_dir, "trainset_28spk", sr, n_fft, hop_length, bucket_sizes)

        train_sampler = BucketSampler(train_dataset.bucket_indices, batch_size=batch_size)
        val_sampler = BucketSampler(val_dataset.bucket_indices, batch_size=batch_size)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)

    elif pad_method == "pto":
        train_dataset = PTODataset(dataset_dir, "trainset_56spk", sr, n_fft, hop_length)
        val_dataset = PTODataset(dataset_dir, "trainset_28spk", sr, n_fft, hop_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pto_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=pto_collate)

    else:
        raise ValueError(f"Invalid PAD_METHOD: {pad_method}\n Choose from ['static', 'dynamic', 'pto']")

    # === Load Model ===
    print(f"Initializing `{model_name}` model.")
    if model_name == "CNN":
        model = CNN()
    elif model_name == "UNet":
        model = UNet()
    elif model_name == "ConvTasNet":
        # ConvTasNet hyperparameters
        model = ConvTasNet(
            enc_dim=enc_dim,
            feature_dim=feature_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            num_stacks=num_stacks,
        )
    else:
        raise ValueError(f"Invalid MODEL: {model_name}\n Choose from ['CNN', 'UNet', 'ConvTasNet']")
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # === Train and Evaluate ===
    val_loss = train_eval(
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.EPOCHS,
        save_pth=config.MODEL_PTH.replace(".pth", f"_{trial.number}.pth"),
        pto=(pad_method == "pto"),
        scheduler=config.SCHEDULER,
    )

    # Return the validation loss for Optuna to minimize
    return val_loss