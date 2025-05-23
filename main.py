import torch
from torch.utils.data import DataLoader
from dataset import HamiltonianGraphDataset, custom_collate
from model import ConditionalGraphGenerator, GraphDiscriminator, get_vit_classifier, get_resnet50_classifier, get_efficientnet_classifier
from train import train_gan, train_combined_cgan, evaluate_model
import torch.optim as optim
import torch.nn as nn
from utils import remove_module_prefix
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import random
from collections import Counter
import os
import multiprocessing


device = torch.device("cuda:2")

torch.cuda.empty_cache()

# # Define the function to sample arbitrary training and validation set sizes
# Define the function to sample arbitrary training and validation set sizes
def sample_train_val(full_dataset, sample_size, test_size=500, val_ratio=0.2, seed=42, pretrain_sample_size=2000):
    """
    Samples datasets for training, validation, and testing, with the option to sample an arbitrary
    number of pretrain samples from the remaining data.
    
    Arguments:
    - full_dataset: The full dataset to sample from.
    - sample_size: The number of data points to use for training and validation combined.
    - test_size: The number of samples to set aside for testing.
    - val_ratio: The ratio of validation samples in the training/validation set.
    - seed: Random seed for reproducibility.
    - pretrain_sample_size: Number of samples to randomly select from the remaining data for pretraining.
                            If None, all remaining data will be used.
                            
    Returns:
    - train_dataset: Subset for training.
    - val_dataset: Subset for validation.
    - test_dataset: Subset for testing.
    - pretrain_dataset: Subset for pretraining.
    """
    # Get the total number of samples
    total_size = len(full_dataset)

    # Ensure that the sample size is valid (i.e., less than the dataset size minus the test set)
    assert sample_size <= total_size - test_size, "Sample size is too large for the available dataset."

    # Separate Hamiltonian and Non-Hamiltonian indices
    hamiltonian_indices = [i for i in range(len(full_dataset)) if full_dataset.labels[i] == 1]
    non_hamiltonian_indices = [i for i in range(len(full_dataset)) if full_dataset.labels[i] == 0]

    # Combine indices and labels
    all_indices = np.array(hamiltonian_indices + non_hamiltonian_indices)
    all_labels = np.array([1] * len(hamiltonian_indices) + [0] * len(non_hamiltonian_indices))  # 1 for Hamiltonian, 0 for Non-Hamiltonian

    # Step 1: Split off the fixed test set (500 samples)
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        all_indices, all_labels, test_size=test_size, stratify=all_labels, random_state=seed
    )

    # Step 2: Sample the desired number of data points from the remaining training/validation set
    sampled_train_val_indices = np.random.choice(train_val_indices, size=sample_size, replace=False)

    # Step 3: Split the sampled data into training (80%) and validation (20%)
    train_size = int(sample_size * (1 - val_ratio))
    train_indices, val_indices = train_test_split(sampled_train_val_indices, test_size=val_ratio, random_state=seed)

    # Create Subsets for training, validation, and testing
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # ---- Step 4: Pretrain dataset should contain all remaining data ----
    remaining_indices = list(set(range(len(full_dataset))) - set(train_indices) - set(val_indices) - set(test_indices))

    # If pretrain_sample_size is None or larger than the remaining data, use all the remaining data
    if pretrain_sample_size is None or pretrain_sample_size > len(remaining_indices):
        pretrain_sample_size = len(remaining_indices)

    # Randomly sample from the remaining data for pretraining
    sampled_pretrain_indices = np.random.choice(remaining_indices, size=pretrain_sample_size, replace=False)
    pretrain_dataset = Subset(full_dataset, sampled_pretrain_indices)

    # ---- Step 5: Print out class distributions ----
    train_labels = [full_dataset.labels[i] for i in train_indices]
    val_labels = [full_dataset.labels[i] for i in val_indices]
    test_labels = [full_dataset.labels[i] for i in test_indices]
    pretrain_labels = [full_dataset.labels[i] for i in sampled_pretrain_indices]

    print("Training set class distribution:", Counter(train_labels))
    print("Validation set class distribution:", Counter(val_labels))
    print("Test set class distribution:", Counter(test_labels))
    print("Pretrain set class distribution:", Counter(pretrain_labels))

    return train_dataset, val_dataset, test_dataset, pretrain_dataset

def my_collate_fn(batch):
    return custom_collate(batch)


def main():
    # Set multiprocessing start method (spawn is required for CUDA)
    multiprocessing.set_start_method('spawn', force=True)
    
    # # Set device (you can change the device index as needed)
    # global device  # Make device accessible by my_collate_fn

    # # Set device
    # device = torch.device("cuda:0")

    # ---- Load Dataset ----
    hamiltonian_dir = './small_planar_dataset/small_nonplanar'
    non_hamiltonian_dir = './small_planar_dataset/small_planar'


    # Create the dataset
    #train_dataset = HamiltonianGraphDataset(hamiltonian_dir, non_hamiltonian_dir)
    full_dataset = HamiltonianGraphDataset(hamiltonian_dir, non_hamiltonian_dir)

    # Define the sample size you want to use (arbitrary number)
    sample_size = 1000
    # Call the function to sample datasets
    train_dataset, val_dataset, test_dataset, pretrain_dataset = sample_train_val(full_dataset, sample_size=sample_size)

    # Create the DataLoader with the custom collate function
    # ---- Create DataLoaders for Train, Validation, and Test ----
    # Create DataLoader with the balanced datasets
    train_dataloader = DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=8,  collate_fn=my_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=36, shuffle=False, num_workers=8,  collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=36, shuffle=False, num_workers=8,  collate_fn=my_collate_fn)
    # DataLoader for pretraining
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=96, shuffle=True, num_workers=8, collate_fn=my_collate_fn)

    # Print the dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Pretrain set size: {len(pretrain_dataset)}")


    # ---- Directly set seeds for reproducibility ----
    seed = 42
    torch.manual_seed(seed)            # PyTorch seed
    torch.cuda.manual_seed(seed)       # CUDA seed (if using GPUs)
    torch.cuda.manual_seed_all(seed)   # If using multi-GPU setups
    np.random.seed(seed)               # NumPy seed
    random.seed(seed)                  # Python random seed

    # Ensure deterministic behavior for some operations in PyTorch's backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # ---- Initialize Models, Optimizers, and Criterion ----
    generator = ConditionalGraphGenerator().to(device)  # Use the conditional generator
    discriminator = GraphDiscriminator().to(device)  # Move model to device
    # classifier = get_resnet50_classifier().to(device)  # Already moved to device in the function
    # classifier = get_efficientnet_classifier().to(device)  # Already moved to device in the function
    classifier = get_vit_classifier().to(device)  # Already moved to device in the function

    # # Use DataParallel to parallelize over multiple GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     generator = nn.DataParallel(generator)
    #     discriminator = nn.DataParallel(discriminator)
    #     #classifier = nn.DataParallel(classifier)

    base_output_dir = "cgan_training_outputs"

    # Ensure the base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    # ---- Pretrain Generator and Discriminator ----
    optimizer_g_pretrain = optim.RMSprop(generator.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizer_d_pretrain = optim.RMSprop(discriminator.parameters(), lr=0.0005, weight_decay=1e-4)

    print("Pretraining Generator and Discriminator...")
    train_gan(generator, discriminator, pretrain_dataloader, optimizer_g_pretrain, optimizer_d_pretrain, device, epochs=500,output_dir=base_output_dir)


    # Define the number of iterations
    num_runs = 3
    base_output_dir = "cgan_training_outputs"

    # Ensure the base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    for run in range(1, num_runs + 1):
        # ---- Load the pretrained models ----
        generator = ConditionalGraphGenerator().to(device)  # Use the conditional generator
        discriminator = GraphDiscriminator().to(device)  # Move model to device
        classifier = get_vit_classifier().to(device)  # Already moved to device in the function
        # classifier = get_resnet50_classifier().to(device)
        # classifier = get_efficientnet_classifier().to(device)
        # Load pretrained generator and discriminator
        try:
            print("Loading pretrained generator and discriminator...")
            generator_pretrained_path = os.path.join(base_output_dir, "best_generator_pretrained.pth")
            pretrained_generator_dict = torch.load(generator_pretrained_path)
            discriminator_pretrained_path = os.path.join(base_output_dir, "best_discriminator_pretrained.pth")
            pretrained_discriminator_dict = torch.load(discriminator_pretrained_path)

            # Remove the 'module.' prefix if loading without DataParallel
            pretrained_generator_dict = remove_module_prefix(pretrained_generator_dict)
            pretrained_discriminator_dict = remove_module_prefix(pretrained_discriminator_dict)

            generator.load_state_dict(pretrained_generator_dict)
            discriminator.load_state_dict(pretrained_discriminator_dict)

            print("Pretrained models loaded successfully.")
        except FileNotFoundError:
            print("Pretrained model files not found. Proceeding without pretraining.")

        # #Use DataParallel to parallelize over multiple GPUs (if available)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     generator = nn.DataParallel(generator)
        #     discriminator = nn.DataParallel(discriminator)
        #     classifier = nn.DataParallel(classifier)

        optimizer_g = optim.RMSprop(generator.parameters(), lr=0.0001, weight_decay=1e-4)
        optimizer_d = optim.RMSprop(discriminator.parameters(), lr=0.0005, weight_decay=1e-4)
        optimizer_c = optim.Adam(classifier.parameters(), lr=0.00001, weight_decay=1e-4)

        # Define learning rate schedulers for the optimizers
        scheduler_g = lr_scheduler.StepLR(optimizer_g, step_size=5, gamma=0.9)
        scheduler_d = lr_scheduler.StepLR(optimizer_d, step_size=5, gamma=0.9)
        scheduler_c = lr_scheduler.StepLR(optimizer_c, step_size=5, gamma=0.9)

        #criterion = nn.BCELoss()  # For GAN
        classification_criterion = nn.CrossEntropyLoss()  # For classification task

        # # ---- Train the framework using Conditional GAN ----
        # train_combined_cgan(generator, discriminator, classifier, 
        #                     train_dataloader, val_dataloader, test_dataloader, 
        #                     optimizer_g, optimizer_d, optimizer_c, scheduler_g, scheduler_d, scheduler_c,
        #                     classification_criterion, device,  # Pass device to the function
        #                     epochs=100)

        # # ---- Test the model ----
        # evaluate_model(generator, classifier, test_dataloader, device)

        # Create a new directory named with the run number (e.g., '1', '2', etc.)
        output_dir = os.path.join(base_output_dir, str(run))
        os.makedirs(output_dir, exist_ok=True)

        print(f"Starting run {run}, outputs will be stored in {output_dir}")

        # ---- Train the framework using Conditional GAN ----
        train_combined_cgan(
            generator, discriminator, classifier,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_g, optimizer_d, optimizer_c,
            scheduler_g, scheduler_d, scheduler_c,
            classification_criterion, device,  # Pass device to the function
            epochs=100,
            output_dir=output_dir
        )

        # ---- Test the model ----
        evaluate_model(generator, classifier, test_dataloader, device)

        print(f"Completed run {run}, results saved in {output_dir}")

if __name__ == '__main__':
    # Set the start method and call main
    multiprocessing.set_start_method('spawn', force=True)
    main()
