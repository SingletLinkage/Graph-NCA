import torch
import traceback
import os
import argparse
from colorama import Fore, Back, Style, init
from src.datasets.Dataset_JPG import Dataset_JPG_Patch
from src.utils.Experiment import Experiment
from src.utils.helper import log_message, load_json_file

# Initialize colorama
init(autoreset=True)

def load_model_config(model_path, log_enabled=True):
    """
    Load model configuration from a specified path.
    Returns the loaded config or None if unsuccessful.
    """
    module = f"{__name__}:load_model_config" if log_enabled else ""
    
    # Check if model_path exists
    if not os.path.exists(model_path):
        log_message(f"Error: Model path does not exist: {model_path}", "ERROR", module, log_enabled)
        return None
        
    # Load the config file
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        log_message(f"Warning: Config file not found at {config_path}", "WARNING", module, log_enabled)
        
        # Attempt to load legacy config file
        config_path = os.path.join(model_path, 'config.dt')
        if not os.path.exists(config_path):
            log_message(f"Error: Legacy config file not found at {config_path}", "ERROR", module, log_enabled)
            return None
        
    # Load the configuration
    config = load_json_file(config_path)

    # Convert input_size from list to tuple if necessary
    keys_to_convert = ['input_size', 'stage1_size', 'stage2_size', 'betas']
    for key in keys_to_convert:
        if isinstance(config[0][key], list):
            config[0][key] = tuple(config[0][key])

    log_message(f"Loaded configuration from {config_path}", "SUCCESS", module, log_enabled, config)
    
    # Update model path in config
    config[0]['model_path'] = model_path
    
    return config

def setup_logging(model_path, log_enabled=True, config=None):
    """
    Setup logging directory and file for the inference run.
    Returns the updated config with logging information.
    """
    module = f"{__name__}:setup_logging" if log_enabled else ""
    
    # Create output directory for inference results
    inference_dir = os.path.join(model_path, "inference_results")
    os.makedirs(inference_dir, exist_ok=True)
    
    if config:
        log_message(f"Inference results will be saved to {inference_dir}", "INFO", module, log_enabled, config)
    
    # Set up logging for this inference run
    from datetime import datetime
    log_dir = os.path.join(model_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    if config:
        config[0]['log_file'] = log_file
        config[0]['verbose_logging'] = log_enabled
    
    with open(log_file, 'w') as f:
        f.write("=== Inference Log ===\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return inference_dir, config

def find_latest_checkpoint(model_path, log_enabled=True, config=None):
    """
    Find the latest model checkpoint in the models directory.
    Returns the path to the latest checkpoint or None if not found.
    """
    module = f"{__name__}:find_latest_checkpoint" if log_enabled else ""
    
    # Find the latest model checkpoint
    models_dir = os.path.join(model_path, 'checkpoints')

    # legacy -> fallback "models" directory
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        log_message(f"Checkpoints directory not found or empty: {models_dir}. Trying legacy 'models' directory.", "WARNING", module, log_enabled, config)
        models_dir = os.path.join(model_path, 'models')

    if not os.path.exists(models_dir):
        log_message(f"Error: Models directory does not exist: {models_dir}", "ERROR", module, log_enabled, config)
        return None
    
    # try to find "best_model.pth" as primary checkpoint
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        log_message(f"Found best model checkpoint: {best_model_path}", "SUCCESS", module, log_enabled, config)
        return best_model_path

    # if not there, try to find "final_model.pth" as secondary checkpoint
    final_model_path = os.path.join(models_dir, 'final_model.pth')
    if os.path.exists(final_model_path):
        log_message(f"Found final model checkpoint: {final_model_path}", "SUCCESS", module, log_enabled, config)
        return final_model_path
    
    # If neither best_model nor final_model exists, look for epoch directories
    log_message("No best or final model checkpoints found. Searching for epoch directories...", "INFO", module, log_enabled, config)
        
    # Find all epoch directories
    epoch_dirs = [d for d in os.listdir(models_dir) if d.startswith('epoch_')]
    if not epoch_dirs:
        log_message("No model checkpoints found in directory", "ERROR", module, log_enabled, config)
        return None
        
    # Sort by epoch number and find the latest checkpoint
    epoch_dirs.sort(key=lambda x: int(x[:-4].split('_')[1]))  # naming convention: epoch_x.pth
    latest_epoch = epoch_dirs[-1]
    latest_model_path = os.path.join(models_dir, latest_epoch)
    
    # Check if the model file exists at this path
    if not os.path.exists(latest_model_path):
        log_message(f"Error: Model checkpoint not found at {latest_model_path}", "ERROR", module, log_enabled, config)
        return None
        
    log_message(f"Found latest model checkpoint: {latest_epoch}", "SUCCESS", module, log_enabled, config)
    return latest_model_path

def initialize_model(config, checkpoint_path, log_enabled=True, model_type='GraphMedNCA'):
    """
    Initialize and load the model based on the configuration.
    Returns the initialized model or None if unsuccessful.
    """
    module = f"{__name__}:initialize_model" if log_enabled else ""
    
    # Define device
    device = torch.device(config[0].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    log_message(f"Using device: {device}", "INFO", module, log_enabled, config)
    log_message("Initializing model...", "INFO", module, log_enabled, config)
    
    # Create the model based on model_type
    if model_type == 'auto':
        model_class = config[0]['model_class']
    elif model_type == 'GraphMedNCA':
        from src.models.GraphMedNCA import GraphMedNCA
        model = GraphMedNCA(
            size=config[0]['input_size'][0],
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'], 
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        from src.agents.Agent_GraphMedNCA import Agent_GraphMedNCA
        agent = Agent_GraphMedNCA(model, log_enabled=log_enabled, config=config)

    elif model_type == 'BasicNCA':
        from src.models.BasicNCA_laplacian import BasicNCA
        model = BasicNCA(
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'], 
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        from src.agents.Agent_MedNCA import Agent_MedNCA
        agent = Agent_MedNCA(model, log_enabled=log_enabled, config=config)

    elif model_type == 'TwoStageNCA':
        from src.models.GraphMedNCA import GraphMedNCA
        from src.models.BackboneNCAB1_changes import BasicNCA
        from src.models.TwoStageNCA import TwoStageNCA
        stage1_model = GraphMedNCA(
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'],
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        stage2_model = BasicNCA(
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'],
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        model = TwoStageNCA(
            stage1_model=stage1_model,
            stage2_model=stage2_model,
            stage1_size=config[0]['stage1_size'],
            stage2_size=config[0]['stage2_size'],
            stage1_steps=config[0]['nca_steps_stage1'],
            stage2_steps=config[0]['nca_steps_stage2'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        from src.agents.Agent_TwoStageNCA import Agent_TwoStageNCA
        agent = Agent_TwoStageNCA(model, log_enabled=log_enabled, config=config)

    else:
        log_message(f"Unsupported model type: {model_type}", "ERROR", module, log_enabled, config)
        return None
    
    log_message("Model and Agent initialized successfully!", "SUCCESS", module, log_enabled, config)
    
    # Load model weights
    try:
        if model_type == 'TwoStageNCA':
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.stage1_model.load_state_dict(checkpoint['stage1_model_state_dict'])
            model.stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])
            # model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.load_state_dict(torch.load(checkpoint_path))
        log_message(f"Successfully loaded model weights from {checkpoint_path}", "SUCCESS", module, log_enabled, config)
    except Exception as e:
        log_message(f"Error loading model weights: {str(e)}", "ERROR", module, log_enabled, config)
        traceback.print_exc()
        return None
    
    return model, agent, device

def run_inference(model, agent, config, inference_dir, log_enabled=True, **kwargs):
    """
    Run inference using the loaded model and configuration.
    Returns the average dice score.
    """
    module = f"{__name__}:run_inference" if log_enabled else ""

    percentage = kwargs.get('percentage', 1.0)
    
    # Initialize the dataset
    log_message("Setting up dataset for inference...", "INFO", module, log_enabled, config)
    dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config, _test=True, percentage=percentage)
        
    # Initialize experiment
    exp = Experiment(config, dataset, model, agent, log_enabled=log_enabled, test_dataset=dataset)
    log_message("Experiment initialized successfully!", "SUCCESS", module, log_enabled, config)
    
    # Set model to evaluation mode
    exp.set_model_state('test')
    dataset.set_experiment(exp, isTest=True)
    
    # Check dataset size
    log_message(f"Dataset size: {len(dataset)}", "INFO", module, log_enabled, config)
    
    if len(dataset) == 0:
        log_message("Error: Dataset is empty. Check image and label paths.", "ERROR", module, log_enabled, config)
        return None
    
    # Run inference
    if log_enabled:
        print(f"\n{Back.BLUE}{Fore.WHITE} RUNNING INFERENCE {Style.RESET_ALL}\n")
    else:
        print("\n=== RUNNING INFERENCE ===\n")
        
    # Run inference with image saving
    dice_score = agent.evaluate_imsave(output_dir=inference_dir, **kwargs)
    
    # Print summary
    log_message(f"Inference complete! Average Dice Score: {dice_score:.4f}", "SUCCESS", module, log_enabled, config)
    log_message(f"Results saved to: {inference_dir}", "INFO", module, log_enabled, config)
    
    return dice_score

def main(model_path, model_type, log_enabled=True, inference_dict=None):
    """
    Main function to run the inference process.
    """
    if inference_dict is None:
        inference_dict = {}

    try:
        module = f"{__name__}:main" if log_enabled else ""
        log_message("=== Initializing inference process ===", "INFO", module, log_enabled)
        
        # Load model configuration
        config = load_model_config(model_path, log_enabled)
        if config is None:
            return
        
        config[0].update(inference_dict)  # Update config with inference parameters
        
        # Setup logging and inference directory
        inference_dir, config = setup_logging(model_path, log_enabled, config)
        
        # Find latest checkpoint
        checkpoint_path = find_latest_checkpoint(model_path, log_enabled, config)
        if checkpoint_path is None:
            return
        
        # Initialize model
        model_result = initialize_model(config, checkpoint_path, log_enabled, model_type)
        if model_result is None:
            return
        model, agent, device = model_result
        
        # Run inference
        dice_score = run_inference(model, agent, config, inference_dir, log_enabled, **inference_dict)
        
        return dice_score
        
    except Exception as e:
        log_message(f"Error in inference execution: {str(e)}", "ERROR", module, log_enabled, config if 'config' in locals() else None)
        traceback.print_exc()
        return None

if __name__ == "__main__":

    main(
        model_path='/home/teaching/group21/final/test-mednca-temp/trial_runs/model_36',
        model_type='BasicNCA',
        log_enabled=True,
        inference_dict={
            'graph_returns': (False),
            'images_to_save': 30,
            'save_graphs': False,
            'percentage': 1,
            'inference_steps': 64,
        }
    )

    # Set up command line argument parsing
    # parser = argparse.ArgumentParser(description='Run inference for MedNCA models')
    # parser.add_argument('--model-path', type=str, 
    #                     default='/home/teaching/group21/final/test-mednca-temp/runs/model_28',
    #                     help='Path to the model directory')
    # parser.add_argument('--log', action='store_true', help='Enable verbose logging with colors')
    # parser.add_argument('--no-log', dest='log', action='store_false', help='Disable verbose logging')
    # parser.add_argument('--save-graphs', action='store_true', help='Save graph visualizations')
    # parser.add_argument('--model-type', type=str, default='GraphMedNCA', 
    #                     choices=['GraphMedNCA', 'BasicNCA'], 
    #                     help='Type of model to use for inference')
    # parser.set_defaults(log=True, save_graphs=False)
    
    # args = parser.parse_args()
    # main(model_path=args.model_path, log_enabled=args.log, save_graphs=args.save_graphs, model_type=args.model_type)

