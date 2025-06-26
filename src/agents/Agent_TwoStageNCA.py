import torch
import os
from src.utils.helper import log_message
from torch.nn import functional as F
import src.agents.agent_helper as helper


class Agent_TwoStageNCA:
    """Agent for managing the two-stage NCA training process"""
    def __init__(self, model, log_enabled=True, config=None):
        self.model = model
        self.config = config[0]  # Extract first config item
        self.device = torch.device(self.config['device'])
        self.log_enabled = log_enabled
        self.best_dice = 0.0
        self.experiment = None
        self.projectConfig = config
        self.nca_steps_stage1 = self.config['nca_steps_stage1']
        self.nca_steps_stage2 = self.config['nca_steps_stage2']
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['lr'],
            betas=self.config['betas']
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config['lr_gamma']
        )
        
        # Loss weights for the two stages
        # TODO: not hardcode these values, make them configurable
        self.stage1_weight = 0.3
        self.stage2_weight = 0.7
        
        self.module_name = f"{self.__class__.__name__}"
        
    def set_exp(self, exp):
        """Set the experiment reference - required by Experiment class"""
        self.experiment = exp
    
    def train(self, data_loader, loss_function):
        self.model.train()
        n_epochs = self.config['n_epoch']
        
        module = f"{self.module_name}:train" if self.log_enabled else ""
        log_message(f"Starting training for {n_epochs} epochs...", "INFO", module, self.log_enabled, config=[self.config])
        
        for epoch in range(n_epochs):
            running_loss = 0.0
            batch_count = 0
            
            # Training loop
            for batch_idx, (_, _, inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass through two-stage model
                outputs, _graph_stuff = self.model(inputs)
                
                # Calculate loss for stage 1 (rough segmentation)
                # Downsample labels to match stage1 output size
                labels_small = F.interpolate(labels, size=outputs['stage1_output'].shape[2:], mode='nearest')
                stage1_loss = loss_function(outputs['stage1_output'], labels_small)
                
                # Calculate loss for stage 2 (refined segmentation)
                stage2_loss = loss_function(outputs['stage2_output'], labels)
                
                # Combined loss with weighting
                loss = self.stage1_weight * stage1_loss + self.stage2_weight * stage2_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    log_message(f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(data_loader)}, "
                                f"Loss: {loss.item():.4f} (Stage1: {stage1_loss.item():.4f}, "
                                f"Stage2: {stage2_loss.item():.4f})", 
                                "INFO", module, self.log_enabled, config=[self.config])
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate average loss for epoch
            epoch_loss = running_loss / batch_count
            log_message(f"Epoch {epoch+1}/{n_epochs} completed, Avg Loss: {epoch_loss:.4f}", 
                        "SUCCESS", module, self.log_enabled, config=[self.config])
            
            # Save model checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # # Evaluate model  
            # if (epoch + 1) % self.config['evaluate_interval'] == 0:
            #     self.evaluate(data_loader, loss_function)
        
            # Save final model at the end of training
            if epoch == n_epochs - 1:
                self.save_checkpoint(epoch, end=True)
    
    def evaluate(self, data_loader, loss_function):
        self.model.eval()
        module = f"{self.module_name}:evaluate" if self.log_enabled else ""
        
        total_stage1_loss = 0.0
        total_stage2_loss = 0.0
        total_combined_loss = 0.0
        batch_count = 0
        
        # Evaluation metrics
        stage1_dice_scores = []
        stage2_dice_scores = []
        
        with torch.no_grad():
            for batch_idx, (_, _, inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs, _graph_stuff = self.model(inputs)
                
                # Calculate loss for stage 1
                labels_small = F.interpolate(labels, size=outputs['stage1_output'].shape[2:], mode='nearest')
                stage1_loss = loss_function(outputs['stage1_output'], labels_small)
                
                # Calculate loss for stage 2
                stage2_loss = loss_function(outputs['stage2_output'], labels)
                
                # Combined loss
                combined_loss = self.stage1_weight * stage1_loss + self.stage2_weight * stage2_loss
                
                total_stage1_loss += stage1_loss.item()
                total_stage2_loss += stage2_loss.item()
                total_combined_loss += combined_loss.item()
                batch_count += 1
                
                # Calculate Dice scores
                stage1_preds = (outputs['stage1_upsampled'] > 0.5).float()
                stage2_preds = (outputs['stage2_output'] > 0.5).float()
                
                for i in range(labels.size(0)):
                    # Calculate Dice coefficient for each sample
                    stage1_dice = self.dice_coefficient(stage1_preds[i], labels[i])
                    stage2_dice = self.dice_coefficient(stage2_preds[i], labels[i])
                    
                    stage1_dice_scores.append(stage1_dice)
                    stage2_dice_scores.append(stage2_dice)
        
        # Calculate average metrics
        avg_stage1_loss = total_stage1_loss / batch_count
        avg_stage2_loss = total_stage2_loss / batch_count
        avg_combined_loss = total_combined_loss / batch_count
        
        avg_stage1_dice = sum(stage1_dice_scores) / len(stage1_dice_scores)
        avg_stage2_dice = sum(stage2_dice_scores) / len(stage2_dice_scores)
        
        log_message(f"Evaluation results:", "INFO", module, self.log_enabled, config=[self.config])
        log_message(f"  Stage 1 - Avg Loss: {avg_stage1_loss:.4f}, Avg Dice: {avg_stage1_dice:.4f}", 
                    "INFO", module, self.log_enabled, config=[self.config])
        log_message(f"  Stage 2 - Avg Loss: {avg_stage2_loss:.4f}, Avg Dice: {avg_stage2_dice:.4f}", 
                    "INFO", module, self.log_enabled, config=[self.config])
        log_message(f"  Combined - Avg Loss: {avg_combined_loss:.4f}", 
                    "INFO", module, self.log_enabled, config=[self.config])
        
        # Save best model
        if avg_stage2_dice > self.best_dice:
            self.best_dice = avg_stage2_dice
            log_message(f"New best Dice score: {self.best_dice:.4f}, saving best model", 
                        "SUCCESS", module, self.log_enabled, config=[self.config])
            self.save_checkpoint(best=True)
        
        self.model.train()
        return avg_stage2_dice
    
    def save_checkpoint(self, epoch=None, best=False, end=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['model_path'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        else:
            if end:
                checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
            else:
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')  
        
        torch.save({
            'stage1_model_state_dict': self.model.stage1_model.state_dict(),
            'stage2_model_state_dict': self.model.stage2_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
        }, checkpoint_path)
        
        module = f"{self.module_name}:save_checkpoint" if self.log_enabled else ""
        log_message(f"Saved checkpoint to {checkpoint_path}", 
                    "INFO", module, self.log_enabled, config=[self.config])
    
    def dice_coefficient(self, pred, target):
        """Calculate Dice coefficient for binary segmentation"""
        smooth = 1e-7
        
        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return dice.item()
    
    def evaluate_imsave(self, output_dir=None, **kwargs):
        return helper.evaluate_imsave(self, output_dir, **kwargs)
        
    
    def getAverageDiceScore_withimsave(self):
        """Calculate average Dice score and save sample predictions"""
        from torch.utils.data import DataLoader
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a data loader with the test dataset
        test_dataset = self.model.test_dataset if hasattr(self.model, 'test_dataset') else None
        
        if test_dataset is None or len(test_dataset) == 0:
            module = f"{self.module_name}:getAverageDiceScore" if self.log_enabled else ""
            log_message(f"No test dataset available, using training dataset", 
                        "WARNING", module, self.log_enabled, config=[self.config])
            # Use a subset of training data
            from torch.utils.data import Subset
            indices = torch.randperm(len(self.model.dataset))[:min(50, len(self.model.dataset))]
            test_dataset = Subset(self.model.dataset, indices)
        
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        self.model.eval()
        dice_scores = []
        
        # Create directory for sample predictions
        samples_dir = os.path.join(self.config['model_path'], 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, (ids, _, inputs, labels) in enumerate(test_loader):
                if i >= 5:  # Limit to 5 batches for saving samples
                    break
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                stage1_output = outputs['stage1_upsampled']
                stage2_output = outputs['stage2_output']


                #thresholding - removed for now
                # Convert to binary predictions
                # stage1_preds = (stage1_output > 0.5).float() #converts to binary mask
                # stage2_preds = (stage2_output > 0.5).float()

                stage1_preds=stage1_output
                stage2_preds=stage2_output

                
                # Calculate Dice scores
                for j in range(inputs.size(0)):
                    dice = self.dice_coefficient(stage2_preds[j], labels[j])
                    dice_scores.append(dice)
                    
                    # Save first 10 samples as images
                    if i * inputs.size(0) + j < 10:
                        self.save_prediction_sample(
                            inputs[j], labels[j], 
                            stage1_preds[j], stage2_preds[j],
                            os.path.join(samples_dir, f"sample_{i * inputs.size(0) + j}.png")
                        )
        
        avg_dice = sum(dice_scores) / len(dice_scores)
        module = f"{self.module_name}:getAverageDiceScore" if self.log_enabled else ""
        log_message(f"Average Dice score on test set: {avg_dice:.4f}", 
                    "INFO", module, self.log_enabled, config=[self.config])
        
        self.model.train()
        return avg_dice
    
    def save_prediction_sample(self, input_img, label, stage1_pred, stage2_pred, save_path):
        """Save a visual comparison of input, label, and predictions"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert tensors to numpy arrays for visualization
        input_np = input_img.cpu().squeeze().numpy()
        label_np = label.cpu().squeeze().numpy()
        stage1_np = stage1_pred.cpu().squeeze().numpy()
        stage2_np = stage2_pred.cpu().squeeze().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Display input image
        axes[0].imshow(input_np, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Display ground truth label
        axes[1].imshow(label_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Display stage 1 prediction
        axes[2].imshow(stage1_np, cmap='gray')
        axes[2].set_title('Stage 1 (GraphMedNCA)')
        axes[2].axis('off')
        
        # Display stage 2 prediction
        axes[3].imshow(stage2_np, cmap='gray')
        axes[3].set_title('Stage 2 (BasicNCA)')
        axes[3].axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()