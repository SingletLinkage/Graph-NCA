import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from colorama import Fore, Back, Style, init
from src.agents.Agent_GraphMedNCA_M1M1 import Agent_EnhancedGraphMedNCA
from src.models.GraphMedNCA_M1M1 import GraphMedNCA # Your GraphMedNCA module
from src.models.GraphMedNCA_M1M1 import EnhancedGraphMedNCA 
from src.datasets.Dataset_JPG import Dataset_JPG_Patch
from src.utils.Experiment import Experiment
from src.utils.helper import log_message
from src.losses.DualBackboneLoss import DualBackboneLoss

# Initialize colorama
init(autoreset=True)

def get_next_run_id(runs_dir):
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        return 0
    
    existing_runs = [d for d in os.listdir(runs_dir) if d.startswith('model_')]
    if not existing_runs:
        return 0
    
    run_numbers = []
    for run_dir in existing_runs:
        try:
            run_num = int(run_dir.split('_')[1])
            run_numbers.append(run_num)
        except (IndexError, ValueError):
            pass
    
    return max(run_numbers) + 1 if run_numbers else 0

def check_directories(img_path, label_path, log_enabled=True):
    """Check if directories exist and contain files"""
    module = f"{__name__}:check_directories" if log_enabled else ""
    
    if not os.path.exists(img_path):
        log_message(f"Error: Image directory does not exist: {img_path}", "ERROR", module, log_enabled)
        return False
        
    if not os.path.exists(label_path):
        log_message(f"Error: Label directory does not exist: {label_path}", "ERROR", module, log_enabled)
        return False
    
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files or not label_files:
        log_message(f"Error: No image or label files found", "ERROR", module, log_enabled)
        return False
    
    log_message(f"Found {len(img_files)} images and {len(label_files)} labels", "SUCCESS", module, log_enabled)
    return True

def test_model_forward(model, sample_data, sample_label, config, device, log_enabled=True):
    """Test model forward pass with enhanced architecture"""
    module = f"{__name__}:test_model_forward" if log_enabled else ""
    
    try:
        log_message("Testing enhanced model forward pass...", "INFO", module, log_enabled)
        
        test_data = sample_data.unsqueeze(0).to(device)
        test_label = sample_label.unsqueeze(0).to(device)
        
        log_message(f"Input shape: {test_data.shape}", "INFO", module, log_enabled)
        log_message(f"Target shape: {test_label.shape}", "INFO", module, log_enabled)
        
        with torch.no_grad():
            outputs = model(test_data, test_label, steps=config['nca_steps'])
            
            log_message(f"Enhanced model test successful!", "SUCCESS", module, log_enabled)
            
            if 'b1_pred' in outputs:
                log_message(f"  - B1 output shape: {outputs['b1_pred'].shape}", "INFO", module, log_enabled)
            if 'b2_pred' in outputs:
                log_message(f"  - B2 output shape: {outputs['b2_pred'].shape}", "INFO", module, log_enabled)
            if 'combined_pred' in outputs:
                log_message(f"  - Combined output shape: {outputs['combined_pred'].shape}", "INFO", module, log_enabled)
                log_message(f"  - Combined output range: [{outputs['combined_pred'].min():.3f}, {outputs['combined_pred'].max():.3f}]", "INFO", module, log_enabled)
            
            # Test loss calculation
            loss_fn = DualBackboneLoss()
            loss, loss_components = loss_fn(outputs, test_label) #basically loss = total_loss
            # loss_components has b1 loss, b2 loss, combined loss but is not used anywhere
            log_message(f"  - Loss calculation successful: {loss:.4f}", "SUCCESS", module, log_enabled)
        
        return True
        
    except Exception as e:
        log_message(f"Error in enhanced model forward test: {e}", "ERROR", module, log_enabled)
        traceback.print_exc()
        return False

# Configuration
config = {
    # Dataset paths
    'img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Training_Input",
    'label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Training_GroundTruth",
    'test_img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Test_Input",
    'test_label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Test_GroundTruth",
    'base_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'),
    'device': "cuda",
    'unlock_CPU': True,

    # Learning parameters
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),

    # Training config
    'save_interval': 5,
    'evaluate_interval': 1,
    'n_epoch': 5,
    'batch_size': 16,

    # Data config
    'input_size': (256, 256),
    'data_split': [1, 0, 0], 
    
    # GraphMedNCA parameters
    'hidden_channels': 16,
    'n_channels': 3,
    'nca_steps': 4,
    'fire_rate': 0.5,
    'patch_size': 64,
    'num_patches': 16,

    # Enhanced loss weights (prioritizes combined output)
    'b1_loss_weight': 0.3,
    'b2_loss_weight': 0.2,
    'combined_loss_weight': 0.5,

    # Logging
    'verbose_logging': True,
}

def main(log_enabled=True):
    try:
        # Setup run directory
        config['run'] = get_next_run_id(config['base_path'])
        config['model_path'] = os.path.join(config['base_path'], f"model_{config['run']}")
        
        # Create log directory
        from datetime import datetime
        log_dir = os.path.join(config['model_path'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d')}.log")
        config['log_file'] = log_file

        # Initialize log file
        with open(log_file, 'w') as f:
            f.write("=== Enhanced GraphMedNCA Dual Backbone Training Log ===\n")
            f.write(f"Run ID: {config['run']}\n")
            f.write(f"Model Path: {config['model_path']}\n")
            f.write(f"NCA Steps: {config['nca_steps']}\n")
            f.write(f"Patch Size: {config['patch_size']}x{config['patch_size']}\n")
            f.write(f"Num Patches: {config['num_patches']}\n")
            f.write(f"Loss Weights - B1: {config['b1_loss_weight']}, B2: {config['b2_loss_weight']}, Combined: {config['combined_loss_weight']}\n")

        module = f"{__name__}:main" if log_enabled else ""
        log_message("=== Initializing Enhanced GraphMedNCA Dual Backbone Training ===", "INFO", module, log_enabled, config=[config])
        
        config['verbose_logging'] = log_enabled
        log_message(f"Next run ID: {config['run']}", "INFO", module, log_enabled, config=[config])
        
        # Check directories
        log_message("Checking data directories...", "INFO", module, log_enabled, config=[config])
        if not check_directories(config['img_path'], config['label_path'], log_enabled):
            return
        if not check_directories(config['test_img_path'], config['test_label_path'], log_enabled):
            return
        
        
        device = torch.device(config['device'])
        # Initialize base GraphMedNCA model
        log_message("Initializing base GraphMedNCA model...", "INFO", module, log_enabled, config=[config])
        base_model = GraphMedNCA(
            hidden_channels=config['hidden_channels'],
            n_channels=config['n_channels'],
            fire_rate=config['fire_rate'],
            device=device,
            log_enabled=log_enabled,
            patch_size=config['patch_size'],
            num_patches=config['num_patches']
        ).to(device)
        log_message("Creating enhanced dual backbone architecture...", "INFO", module, log_enabled, config=[config])
        model = EnhancedGraphMedNCA(base_model).to(device)
        # Initialize enhanced loss function and optimizer
        log_message("Setting up enhanced loss function and optimizer...", "INFO", module, log_enabled, config=[config])
        loss_fn = DualBackboneLoss(
            b1_weight=config['b1_loss_weight'], 
            b2_weight=config['b2_loss_weight'],
            combined_weight=config['combined_loss_weight']
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            betas=config['betas']
        )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=config['lr_gamma']
        )
        
        log_message("Enhanced optimizer and scheduler initialized", "SUCCESS", module, log_enabled, config=[config])
        # agent = Agent_GraphMedNCA(model=base_model, log_enabled=log_enabled, config=[config])
        agent = Agent_EnhancedGraphMedNCA(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            log_enabled=log_enabled
        )
        # Setup datasets

        log_message("Setting up datasets...", "INFO", module, log_enabled, config=[config])
        dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=[config])
        test_dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=[config], _test=True)
        experiment = Experiment(config=[config], dataset=dataset, model=None, agent=agent, log_enabled=log_enabled, test_dataset=test_dataset)
        dataset.set_experiment(experiment)
        test_dataset.set_experiment(experiment, isTest=True)
        

        

        # Wrap with enhanced architecture
        
        
        log_message(f"Enhanced GraphMedNCA model initialized successfully", "SUCCESS", module, log_enabled, config=[config])

        # Test model with sample data
        log_message("Testing enhanced model with sample data...", "INFO", module, log_enabled, config=[config])
        _, sample_data,_, sample_label = dataset[0]
        sample_data = sample_data.to(device)
        sample_label = sample_label.to(device)
        
        if not test_model_forward(model, sample_data, sample_label, config, device, log_enabled):
            log_message("Enhanced model test failed, aborting training", "ERROR", module, log_enabled, config=[config])
            return
        
        
        # Create data loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        log_message(f"Data loaders created - Train: {len(train_loader)}, Test: {len(test_loader)}", "SUCCESS", module, log_enabled, config=[config])

        # Training loop
        log_message("Starting enhanced training...", "INFO", module, log_enabled, config=[config])
        
        best_loss = float('inf')
        best_combined_loss = float('inf')
        training_metrics = {
            'epoch_losses': [],
            'b1_losses': [],
            'b2_losses': [],
            'combined_losses': [],
            'consistency_losses': [],
            'learning_rates': []
        }


        agent.train(train_loader, test_loader, training_metrics, config)

        
    except Exception as e:
        log_message(f"Critical error in enhanced training loop: {e}", "ERROR", module, log_enabled, config=[config])
        traceback.print_exc()
        raise e
        
    #     for epoch in range(config['n_epoch']):
    #         agent.train_one_epoch(train_loader, epoch, training_metrics)
    #         if (epoch + 1) % config['evaluate_interval'] == 0:
    #         avg_epoch_loss, avg_combined_loss = agent.evaluate(test_loader)

    #         # Save best model based on combined loss (primary metric)
    #         current_primary_loss = avg_combined_loss if avg_combined_loss > 0 else avg_epoch_loss
    #         if current_primary_loss < best_combined_loss:
    #             best_combined_loss = current_primary_loss
    #             best_loss = avg_epoch_loss
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'total_loss': best_loss,
    #                 'combined_loss': best_combined_loss,
    #                 'config': config,
    #                 'training_metrics': training_metrics
    #             }, os.path.join(config['model_path'], 'best_model.pth'))
    #             log_message(f"  - New best model saved (combined loss: {best_combined_loss:.4f})", "SUCCESS", module, log_enabled, config=[config])
    #             model_path = config['model_path']
    #             agent.save_checkpoint(model_path, epoch, training_metrics, best=True)
    #             agent.save_best_epoch_visualizations(model, test_loader, device, config['model_path'], config, log_enabled)
    #         agent.save_checkpoint(model_path, epoch, training_metrics, best=False)

    #         # Evaluation
    #         if (epoch + 1) % config['evaluate_interval'] == 0:
    #             log_message("Running evaluation...", "INFO", module, log_enabled, config=[config])
    #             model.eval()
    #             eval_loss = 0.0
    #             eval_combined_loss = 0.0
    #             eval_batches = 0

    #             with torch.no_grad():
    #                 for _, _, data, target in test_loader:
    #                     data, target = data.to(device), target.to(device)
    #                     outputs = model(data, target, steps=config['nca_steps'])
    #                     loss, loss_components = loss_fn(outputs , target)
    #                     eval_loss += loss_components['total_loss']
    #                     if 'combined_loss' in loss_components:
    #                         eval_combined_loss += loss_components['combined_loss']
    #                     eval_batches += 1
                        
    #                     if eval_batches >= 20:  # Limit evaluation batches
    #                         break

    #             if eval_batches > 0:
    #                 avg_eval_loss = eval_loss / eval_batches
    #                 avg_eval_combined = eval_combined_loss / eval_batches if eval_combined_loss > 0 else 0
    #                 log_message(f"  - Evaluation Total Loss: {avg_eval_loss:.4f}", "INFO", module, log_enabled, config=[config])
    #                 if avg_eval_combined > 0:
    #                     log_message(f"  - Evaluation Combined Loss: {avg_eval_combined:.4f}", "INFO", module, log_enabled, config=[config])

    #             model.train()

    #         Update learning rate
    #         scheduler.step()
        
    #     # Final save
    #     torch.save({
    #         'epoch': config['n_epoch'],
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'total_loss': training_metrics['epoch_losses'][-1] if training_metrics['epoch_losses'] else 0,
    #         'combined_loss': training_metrics['combined_losses'][-1] if training_metrics['combined_losses'] else 0,
    #         'config': config,
    #         'training_metrics': training_metrics
    #     }, os.path.join(config['model_path'], 'final_model.pth'))
        
    #     log_message("=== Enhanced Training Completed Successfully ===", "SUCCESS", module, log_enabled, config=[config])
    #     log_message(f"Best total loss achieved: {best_loss:.4f}", "INFO", module, log_enabled, config=[config])
    #     log_message(f"Best combined loss achieved: {best_combined_loss:.4f}", "INFO", module, log_enabled, config=[config])
    #     log_message(f"Model saved in: {config['model_path']}", "INFO", module, log_enabled, config=[config])
        
    #     # Log final metrics to file
    #     with open(config['log_file'], 'a') as f:
    #         f.write(f"\n=== Enhanced Training Completed ===\n")
    #         f.write(f"Best Total Loss: {best_loss:.4f}\n")
    #         f.write(f"Best Combined Loss: {best_combined_loss:.4f}\n")
    #         f.write(f"Final B1 Loss: {training_metrics['b1_losses'][-1] if training_metrics['b1_losses'] else 0:.4f}\n")
    #         f.write(f"Final B2 Loss: {training_metrics['b2_losses'][-1] if training_metrics['b2_losses'] else 0:.4f}\n")
    #         f.write(f"Final Combined Loss: {training_metrics['combined_losses'][-1] if training_metrics['combined_losses'] else 0:.4f}\n")
    #         f.write(f"Final Consistency Loss: {training_metrics['consistency_losses'][-1] if training_metrics['consistency_losses'] else 0:.4f}\n")
    #         f.write(f"Epochs Completed: {config['n_epoch']}\n")
        
    # except Exception as e:
    #     log_message(f"Critical error in enhanced training loop: {e}", "ERROR", module, log_enabled, config=[config])
    #     traceback.print_exc()
    #     raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Enhanced GraphMedNCA Dual Backbone')
    parser.add_argument('--quiet', action='store_true', help='Disable verbose logging')
    args = parser.parse_args()
    
    log_enabled = not args.quiet
    main(log_enabled=log_enabled)
