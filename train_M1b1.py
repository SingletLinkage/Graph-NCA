import torch
import traceback
import os
import argparse
from colorama import Fore, Back, Style, init
from src.models.GraphMedNCA import GraphMedNCA
from src.models.BackboneNCAB1 import BasicNCA
from src.datasets.Dataset_JPG import Dataset_JPG_Patch
from src.utils.Experiment import Experiment
from src.losses.LossFunctions import DiceBCELoss
from src.utils.helper import log_message
from src.models.TwoStageNCA import TwoStageNCA
from src.agents.Agent_TwoStageNCA import Agent_TwoStageNCA


# Initialize colorama
init(autoreset=True)

def get_next_run_id(runs_dir):
    """
    Check if model_path and 'runs' subdirectory exist, create if needed.
    If it exists, determine the next run ID based on existing directories.
    """
    
    # Create directories if they don't exist
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        return 0
    
    # Get existing run directories
    existing_runs = [d for d in os.listdir(runs_dir) if d.startswith('model_')]
    
    if not existing_runs:
        return 0
    
    # Extract run numbers and find the highest
    run_numbers = []
    for run_dir in existing_runs:
        try:
            run_num = int(run_dir.split('_')[1])
            run_numbers.append(run_num)
        except (IndexError, ValueError):
            pass
    
    return max(run_numbers) + 1 if run_numbers else 0


config = [{
    'img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Training_Input",
    'label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Training_GroundTruth",
    'test_img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Test_Input",
    'test_label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Test_GroundTruth",
    'base_path': os.path.join(os.getcwd(), 'runs'),
    'device': "cuda",
    'unlock_CPU': True,

    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),

    'save_interval': 5,
    'evaluate_interval': 1,
    'n_epoch': 25,
    'batch_size': 32,  # Reduced slightly due to increased memory requirements

    'stage1_size': (64, 64),    # Low resolution for GraphMedNCA
    'stage2_size': (256, 256),  # High resolution for BasicNCA
    'input_size': (256, 256),   # Dataset will load images at this resolution
    'data_split': [1, 0, 0],

    # Model parameters - using separate parameters for each stage
    # Stage 1 (GraphMedNCA)
    'hidden_channels': 32,
    'nca_steps': 8,
    'fire_rate': 0.5,

    # For Dataset
    'input_channels': 1,
    'output_channels': 1,

    # Logging configuration
    'verbose_logging': True,
    'inference_steps': 1,
}]
# TODO: add specific parameters for each stage

def check_image_label_directories(img_path, label_path, log_enabled=True):
    """Check if the image and label directories exist and contain matching files"""
    module = f"{__name__}:check_image_label_directories" if log_enabled else ""
    
    # Check if directories exist
    if not os.path.exists(img_path):
        log_message(f"Error: Image directory does not exist: {img_path}", "ERROR", module, log_enabled, config=config)
        return False
        
    if not os.path.exists(label_path):
        log_message(f"Error: Label directory does not exist: {label_path}", "ERROR", module, log_enabled, config=config)
        return False
    
    # Get files from both directories
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        log_message(f"Error: No image files found in {img_path}", "ERROR", module, log_enabled, config=config)
        return False
        
    if not label_files:
        log_message(f"Error: No label files found in {label_path}", "ERROR", module, log_enabled, config=config)
        # Print a sample of what files are there
        all_files = os.listdir(label_path)
        log_message(f"Files in label directory: {all_files[:10]}", "WARNING", module, log_enabled, config=config)
        return False
    
    # Check if there are matching files
    common_files = set(img_files).intersection(set(label_files))
    if not common_files:
        log_message(f"Warning: No matching filenames between image and label directories", "WARNING", module, log_enabled, config=config)
        log_message(f"Image files: {img_files[:5]}", "WARNING", module, log_enabled, config=config)
        log_message(f"Label files: {label_files[:5]}", "WARNING", module, log_enabled, config=config)
        # Note: We'll continue anyway since the Dataset_JPG_Patch class seems to handle this
    else:
        log_message(f"Found {len(common_files)} matching files between image and label directories", "SUCCESS", module, log_enabled, config=config)
    return True


def main(log_enabled=True):
    try:
        # Set model path and run ID
        config[0]['run'] = get_next_run_id(config[0]['base_path'])
        config[0]['model_path'] = os.path.join(config[0]['base_path'], f"model_{config[0]['run']}")

        # Set up logging
        from datetime import datetime
        log_dir = os.path.join(config[0]['model_path'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d')}.log")
        config[0]['log_file'] = log_file

        with open(log_file, 'w') as f:
            f.write("=== Two-Stage Med-NCA Run Log ===\n")
            f.write(f"Run ID: {config[0]['run']}\n")
            f.write(f"Model Path: {config[0]['model_path']}\n")
            f.write(f"Stage 1 size: {config[0]['stage1_size']} (GraphMedNCA)\n")
            f.write(f"Stage 2 size: {config[0]['stage2_size']} (BasicNCA)\n")
            f.write(f"Image path: {config[0]['img_path']}\n")
            f.write(f"Label path: {config[0]['label_path']}\n")

        module = f"{__name__}:main" if log_enabled else ""
        log_message("=== Initializing two-stage training process ===", "INFO", module, log_enabled, config=config)
        
        # Update config with logging preference
        config[0]['verbose_logging'] = log_enabled

        # Check directories - same as in original code
        img_path = config[0]['img_path']
        label_path = config[0]['label_path']
        test_img_path = config[0].get('test_img_path', None)
        test_label_path = config[0].get('test_label_path', None)
        
        log_message("Checking data directories...", "INFO", module, log_enabled, config=config)
        check_image_label_directories(img_path, label_path, log_enabled)
        check_image_label_directories(test_img_path, test_label_path, log_enabled)
        
        # Set up datasets - using stage2_size as the input size
        log_message("Setting up dataset...", "INFO", module, log_enabled, config=config)
        dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config)
        test_dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config, _test=True)
        
        # Set up device
        device = torch.device(config[0]['device'])
        
        # Initialize models for the two stages
        log_message("Initializing models for both stages...", "INFO", module, log_enabled, config=config)
        
        # Stage 1: GraphMedNCA for low-res rough segmentation
        stage1_model = GraphMedNCA(
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'],
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        log_message("Stage 1 GraphMedNCA initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Stage 2: BasicNCA for refinement - since we're following the figure, 
        # the BasicNCA operates on the output of the GraphMedNCA directly (not concatenated)
        stage2_model = BasicNCA(
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'],
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        log_message("Stage 2 BasicNCA initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Combine models into TwoStageNCA
        combined_model = TwoStageNCA(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            stage1_size=config[0]['stage1_size'],
            stage2_size=config[0]['stage2_size'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        log_message("Two-stage model constructed successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Create TwoStageNCA agent
        agent = Agent_TwoStageNCA(combined_model, log_enabled=log_enabled, config=config)
        log_message("Two-stage agent initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Create experiment
        exp = Experiment(config, dataset, combined_model, agent, log_enabled=log_enabled, test_dataset=test_dataset)
        log_message("Experiment initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        
        exp.set_model_state('train')
        dataset.set_experiment(exp)  
        test_dataset.set_experiment(exp, isTest=True)
        
        # Attach datasets to model for easier access during evaluation
        combined_model.dataset = dataset
        combined_model.test_dataset = test_dataset
        
        # Print dataset info
        log_message(f"Dataset size: {len(dataset)}", "INFO", module, log_enabled, config=config)
        log_message(f"Test dataset size: {len(test_dataset)}", "INFO", module, log_enabled, config=config)
        
        # Skip training if dataset is empty
        if len(dataset) == 0:
            log_message("Error: Dataset is empty. Check image and label paths.", "ERROR", module, log_enabled, config=config)
            return
        
        # Test loading a sample
        log_message("Testing dataset sample loading and model forward pass...", "INFO", module, log_enabled, config=config)
        try:
            sample_id, sample_col_data, sample_data, sample_label = dataset[0]
            log_message(f"Successfully loaded sample: data shape {sample_data.shape}, label shape {sample_label.shape}", 
                       "SUCCESS", module, log_enabled, config=config)
            
            # Test model forward pass
            with torch.no_grad():
                test_output = combined_model(sample_data.unsqueeze(0).to(device))
                log_message(f"Forward pass successful!", "SUCCESS", module, log_enabled, config=config)
                log_message(f"  Stage 1 output shape: {test_output['stage1_output'].shape}", 
                           "SUCCESS", module, log_enabled, config=config)
                log_message(f"  Stage 1 upsampled shape: {test_output['stage1_upsampled'].shape}", 
                           "SUCCESS", module, log_enabled, config=config)
                log_message(f"  Stage 2 output shape: {test_output['stage2_output'].shape}", 
                           "SUCCESS", module, log_enabled, config=config)
        except Exception as e:
            log_message(f"Error testing dataset: {e}", "ERROR", module, log_enabled, config=config)
            traceback.print_exc()
            return
        
        # Create data loader
        log_message("Creating data loader...", "INFO", module, log_enabled, config=config)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            shuffle=True, 
            batch_size=config[0]['batch_size'],
            num_workers=0
        )
        
        # Initialize loss function
        loss_function = DiceBCELoss()
        
        # Print model parameters
        stage1_params = sum(p.numel() for p in stage1_model.parameters() if p.requires_grad)
        stage2_params = sum(p.numel() for p in stage2_model.parameters() if p.requires_grad)
        total_params = stage1_params + stage2_params
        
        log_message(f"Stage 1 model parameters: {stage1_params}", "INFO", module, log_enabled, config=config)
        log_message(f"Stage 2 model parameters: {stage2_params}", "INFO", module, log_enabled, config=config)
        log_message(f"Total model parameters: {total_params}", "INFO", module, log_enabled, config=config)
        
        # Train model
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} STARTING TWO-STAGE TRAINING {Style.RESET_ALL}\n")
        else:
            print("\n=== STARTING TWO-STAGE TRAINING ===\n")
        
        agent.train(data_loader, loss_function)
        log_message("Training completed successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Evaluate model
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} EVALUATING TWO-STAGE MODEL {Style.RESET_ALL}\n")
        else:
            print("\n=== EVALUATING TWO-STAGE MODEL ===\n")
        
        dice_score = agent.getAverageDiceScore_withimsave()
        
        if log_enabled:
            log_message(f"Evaluation complete! Average Dice Score: {dice_score}", "SUCCESS", module, log_enabled, config=config)
        else:
            print(f"Evaluation complete! Average Dice Score: {dice_score:.3f}")
        
    except Exception as e:
        log_message(f"Error in main execution: {str(e)}", "ERROR", module, log_enabled, config=config)
        traceback.print_exc()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train Two-Stage GraphMedNCA + BasicNCA model')
    parser.add_argument('--log', action='store_true', help='Enable verbose logging with colors')
    parser.add_argument('--no-log', dest='log', action='store_false', help='Disable verbose logging')
    parser.set_defaults(log=True)
    
    args = parser.parse_args()
    main(log_enabled=args.log)