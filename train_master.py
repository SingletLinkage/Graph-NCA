import torch
import traceback
import os
import argparse
from colorama import Fore, Back, Style, init
from src.utils.Experiment import Experiment
from src.utils.helper import log_message

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

def check_image_label_directories(img_path, label_path, config, log_enabled=True):
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
    return True

def init_logging(config):
    config[0]['run'] = get_next_run_id(config[0]['base_path'])
    config[0]['model_path'] = os.path.join(config[0]['base_path'], f"model_{config[0]['run']}")

    from datetime import datetime
    log_dir = os.path.join(config[0]['model_path'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d')}.log")
    config[0]['log_file'] = log_file

    with open(log_file, 'w') as f:
        f.write("=== Run Log ===\n")
        f.write(f"Run ID: {config[0]['run']}\n")
        f.write(f"Model Path: {config[0]['model_path']}\n")
        f.write(f"nca_steps: {config[0]['nca_steps_stage1']}\n")
        f.write(f"nca_steps: {config[0]['nca_steps_stage2']}\n")
        f.write(f"image_path: {config[0]['img_path']}\n")
        f.write(f"label_path: {config[0]['label_path']}\n")

def get_config(is_trial=True, **kwargs):
    """
    Load configuration from config.json file.
    If the file does not exist, create a default configuration.
    """
    import json
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if not os.path.exists(config_path):
        # Create default config if it doesn't exist
        config = [{
            # 'img_path': r"/home/teaching/group21/Dataset/2016_sample_img/2016_actual/2016_actual",
            # 'label_path': r"/home/teaching/group21/Dataset/2016_sample_img/2016_masked/2016_masked",
            'img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Training_Input",
            'label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Training_GroundTruth",
            'test_img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Test_Input",
            'test_label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Test_GroundTruth",
            'base_path': os.path.join(os.path.dirname(__file__), 'runs'),
            'device': "cuda",  # Use CPU for now for stability - change to cuda if needed
            'unlock_CPU': True,  # Add this to avoid thread limitation

            # Learning rate
            'lr': 1e-4,
            'lr_gamma': 0.9999,
            'betas': (0.9, 0.99),

            # Training config
            'save_interval': 10,
            'evaluate_interval': 10,
            'n_epoch': 40, 
            'batch_size': 8, 

            # Data
            # 'input_size': (64, 64), --> one stage config
            # 'data_split': [0.7, 0, 0.3], 
            'data_split': [1, 0, 0], 
            # 'test_split': [0, 0, 1],

            # two - stage config
            'stage1_size': (64, 64),    # Low resolution for GraphMedNCA
            'stage2_size': (256, 256),  # High resolution for BasicNCA
            'input_size': (256, 256),   # Dataset will load images at this resolution

            # Graph-NCA parameters
            'hidden_channels': 32,  # Reduced for faster training during debugging
            'nca_steps': 10, #useless for now
            'nca_steps_stage1': 4,  # Number of steps for the first stage NCA
            'nca_steps_stage2': 10,  # Number of steps for the second stage NCA 
            # kept stage2 steps more cuz we observed that in stage1, it was able to learn with fewer steps
            'fire_rate': 0.5,

            # For JPG dataset
            'input_channels': 1,
            'output_channels': 1,

            # Logging configuration
            'verbose_logging': True,
            'inference_steps' : 10, 
        }]
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Update config with any additional kwargs
    for key, value in kwargs.items():
        if isinstance(config, list) and len(config) > 0:
            config[0][key] = value
        else:
            config[key] = value

    if is_trial:
        config[0]['base_path'] = os.path.join(os.getcwd(), 'trial_runs')

    return config

def train(config, nca, agent, dataset, test_dataset, loss_function, device, save_graphs=None, log_enabled=True, evaluation_dict=None):
    try:
        if evaluation_dict is None:
            evaluation_dict = {}

        # save agent and nca modules in config - for later access
        config[0]['agent_module'] = agent.__module__
        config[0]['agent_class'] = agent.__class__.__name__
        config[0]['model_module'] = nca.__module__
        config[0]['model_class'] = nca.__class__.__name__

        if hasattr(nca, 'stage1_model'):
            config[0]['stage1_model'] = nca.stage1_model.__module__
            config[0]['stage1_model_class'] = nca.stage1_model.__class__.__name__
        if hasattr(nca, 'stage2_model'):
            config[0]['stage2_model'] = nca.stage2_model.__module__
            config[0]['stage2_model_class'] = nca.stage2_model.__class__.__name__
            

        module = f"{__name__}:train" if log_enabled else ""
        log_message("=== Initializing training process ===", "INFO", module, log_enabled, config=config)
        
        # Update config with logging preference
        config[0]['verbose_logging'] = log_enabled

        log_message(f"Next run ID: {config[0]['run']}", "INFO", module, log_enabled, config=config)
        
        # # Make sure the Dataset_JPG_Patch.py file exists and is importable
        # patch_file = os.path.join(os.path.dirname(__file__), 'src', 'datasets', 'Dataset_JPG_Patch.py')
        # if not os.path.exists(patch_file):
        #     log_message(f"Creating patched dataset file at {patch_file}...", "WARNING", module, log_enabled, config=config)
        #     # Create directory if it doesn't exist
        #     os.makedirs(os.path.dirname(patch_file), exist_ok=True)
        
        # Check image and label directories before creating dataset
        img_path = config[0]['img_path']
        label_path = config[0]['label_path']

        test_img_path = config[0].get('test_img_path', None)
        test_label_path = config[0].get('test_label_path', None)
        
        log_message("Checking data directories...", "INFO", module, log_enabled, config=config)
        check_image_label_directories(img_path, label_path, config, log_enabled)
        check_image_label_directories(test_img_path, test_label_path, config, log_enabled)
        
        exp = Experiment(config, dataset, nca, agent, log_enabled=log_enabled, test_dataset=test_dataset)
        log_message("Experiment initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        exp.set_model_state('train')
        dataset.set_experiment(exp)  # This should now initialize the dataset properly
        test_dataset.set_experiment(exp, isTest=True)  # Set up test dataset as well
        
        # Now we can check the dataset info
        log_message(f"Dataset size: {len(dataset)}", "INFO", module, log_enabled, config=config)
        log_message(f"Test dataset size: {len(test_dataset)}", "INFO", module, log_enabled, config=config)
        
        # Skip training if dataset is empty
        if len(dataset) == 0:
            log_message("Error: Dataset is empty. Check image and label paths.", "ERROR", module, log_enabled, config=config)
            return
            
        # Test loading a sample from the dataset
        log_message("Testing dataset sample loading...", "INFO", module, log_enabled, config=config)
        try:
            sample_id, sample_col_data, sample_data, sample_label = dataset[0]
            log_message(f"Successfully loaded sample: data shape {sample_data.shape}, label shape {sample_label.shape}", "SUCCESS", module, log_enabled, config=config)
            
            # Test model forward pass on a single sample
            log_message("Testing model forward pass...", "INFO", module, log_enabled, config=config)
            with torch.no_grad():
                test_output = nca(sample_data.unsqueeze(0).to(device))
                if isinstance(test_output, dict):
                    for key, value in test_output.items():
                        log_message(f"Forward pass output for {key}: shape {value.shape}", "SUCCESS", module, log_enabled, config=config)
                else:
                    log_message(f"Forward pass successful!", "SUCCESS", module, log_enabled, config=config)
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
            num_workers=0  # Use 0 workers for debugging
        )        
        # Print model parameters
        log_message(f"Model parameters: {sum(p.numel() for p in nca.parameters() if p.requires_grad)}", "INFO", module, log_enabled, config=config)
        
        # Train model
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} STARTING TRAINING {Style.RESET_ALL}\n")
        else:
            print("\n=== STARTING TRAINING ===\n")
            
        agent.train(data_loader, loss_function)
        log_message("Training completed successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Evaluate model
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} EVALUATING MODEL {Style.RESET_ALL}\n")
        else:
            print("\n=== EVALUATING MODEL ===\n")

        if save_graphs is not None:
            dice_score = agent.evaluate_imsave(saveGraphs=save_graphs, **evaluation_dict)
        else:
            dice_score = agent.evaluate_imsave(**evaluation_dict)

        if log_enabled:
            log_message(f"Evaluation complete! Average Dice Score: {dice_score}", "SUCCESS", module, log_enabled, config=config)
        else:
            print(f"Evaluation complete! Average Dice Score: {dice_score:.3f}")
        
    except Exception as e:
        log_message(f"Error in main execution: {str(e)}", "ERROR", module, log_enabled, config=config)
        traceback.print_exc()


def main(log_enabled=True):
    module = f"{__name__}:main" if log_enabled else ""

    config = get_config(
        is_trial=False,
        n_epoch=50,
        batch_size=16,
    )

    init_logging(config)

    # Use patched Dataset_JPG with resize=True
    log_message("Setting up dataset...", "INFO", module, log_enabled, config=config)
    from src.datasets.Dataset_JPG import Dataset_JPG_Patch
    dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config, percentage=1)
    test_dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config, _test=True, percentage=1)
    device = torch.device(config[0]['device'])


    # # Create the experiment first to set up the dataset
    # from src.models.GraphMedNCA import GraphMedNCA
    # log_message("Initializing model...", "INFO", module, log_enabled, config=config)
    # nca = GraphMedNCA(
    #     size=config[0]['input_size'][0],
    #     hidden_channels=config[0]['hidden_channels'],
    #     n_channels=config[0]['input_channels'], 
    #     fire_rate=config[0]['fire_rate'],
    #     device=device,
    #     log_enabled=log_enabled
    # ).to(device)
    # log_message("GraphMedNCA Model initialized successfully!", "SUCCESS", module, log_enabled, config=config)

    # from src.agents.Agent_GraphMedNCA import Agent_GraphMedNCA
    # agent = Agent_GraphMedNCA(nca, log_enabled=log_enabled, config=config)
    # log_message("Agent_GraphMedNCA initialized successfully!", "SUCCESS", module, log_enabled, config=config)


    # from src.models.BasicNCA_laplacian import BasicNCA
    # nca = BasicNCA(
    #         hidden_channels=config[0]['hidden_channels'],
    #         n_channels=config[0]['input_channels'], 
    #         fire_rate=config[0]['fire_rate'],
    #         device=device,
    #         log_enabled=log_enabled
    # ).to(device)
    # log_message("BasicNCA Model initialized successfully!", "SUCCESS", module, log_enabled, config=config)
    
    # from src.agents.Agent_MedNCA import Agent_MedNCA
    # agent = Agent_MedNCA(nca, log_enabled=log_enabled, config=config)
    # log_message("Agent_MedNCA initialized successfully!", "SUCCESS", module, log_enabled, config=config)


    # Stage 1: GraphMedNCA for low-res rough segmentation
    from src.models.BackboneNCAB1_changes import BasicNCA
    from src.models.GraphMedNCA import GraphMedNCA
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
    from src.models.TwoStageNCA import TwoStageNCA
    nca = TwoStageNCA(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage1_size=config[0]['stage1_size'],
        stage2_size=config[0]['stage2_size'],
        stage1_steps=config[0]['nca_steps_stage1'],
        stage2_steps=config[0]['nca_steps_stage2'],
        device=device,
        log_enabled=log_enabled
    ).to(device)
    log_message("Two-stage model constructed successfully!", "SUCCESS", module, log_enabled, config=config)
    
    # Create TwoStageNCA agent
    from src.agents.Agent_TwoStageNCA import Agent_TwoStageNCA
    agent = Agent_TwoStageNCA(nca, log_enabled=log_enabled, config=config)
    log_message("Two-stage agent initialized successfully!", "SUCCESS", module, log_enabled, config=config)

    from src.losses.LossFunctions import DiceBCELoss
    loss_function = DiceBCELoss()

    train(
        config=config, 
        nca=nca,
        agent=agent, 
        dataset=dataset, 
        test_dataset=test_dataset, 
        loss_function=loss_function,
        device=device,
        log_enabled=log_enabled,
        evaluation_dict = {
            'threshold': None,
            'graph_returns': (True, False),
            'save_graphs': True,  # -> if you want to save graph data, set this to True
            'images_to_save': 'all',  # any non-numerical value will save all images
        }
    )
    

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train Graph-based MedNCA model')
    parser.add_argument('--log', action='store_true', help='Enable verbose logging with colors')
    parser.add_argument('--no-log', dest='log', action='store_false', help='Disable verbose logging')
    parser.set_defaults(log=True)
    
    args = parser.parse_args()
    main(log_enabled=args.log)