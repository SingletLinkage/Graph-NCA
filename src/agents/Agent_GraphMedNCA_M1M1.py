import os
import torch
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np

from src.utils.helper import log_message

class Agent_EnhancedGraphMedNCA:
    def __init__(self, model, loss_fn=None, optimizer=None, scheduler=None, device=None, config=None, log_enabled=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.log_enabled = log_enabled
        self.experiment = None  # Will be set by set_exp
    def set_exp(self, experiment):
        """
        Set the experiment object for this agent.
        This method is called by the Experiment class.
        """
        self.experiment = experiment

        # Get logging preference from experiment config if available
        if experiment.get_from_config('verbose_logging') is not None:
            self.log_enabled = experiment.get_from_config('verbose_logging')

        # Initialize optimizer after experiment is set, if not already set
        self.setup_optimizer()
    def setup_optimizer(self):
        """
        Initialize optimizer using experiment config if not already set.
        Should be called after experiment is set.
        """
        if self.experiment is None:
            return

        # Only set optimizer if not already set
        if self.optimizer is None:
            lr = self.experiment.get_from_config('lr')
            betas = self.experiment.get_from_config('betas')
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(lr),
                betas=betas
            )
            module = f"{__name__}" if self.log_enabled else ""
            log_message(f"Optimizer initialized with lr={lr}, betas={betas}", "SUCCESS", module, self.log_enabled, [self.config])

        # Only set scheduler if not already set
        if self.scheduler is None:
            lr_gamma = self.experiment.get_from_config('lr_gamma')
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=lr_gamma
            )
            module = f"{__name__}" if self.log_enabled else ""
            log_message(f"Scheduler initialized with gamma={lr_gamma}", "SUCCESS", module, self.log_enabled, [self.config])

    def train(self, train_loader, test_loader, training_metrics, config):
        best_loss = float('inf')
        model_path = config['model_path']
        os.makedirs(model_path, exist_ok=True)
        n_epoch = config['n_epoch']

        for epoch in range(n_epoch):
            self.train_one_epoch(train_loader, epoch, training_metrics)
            avg_eval_loss, avg_eval_combined = self.evaluate(test_loader)
            # Save best model
            if avg_eval_combined < best_loss:
                best_loss = avg_eval_combined
                self.save_checkpoint(model_path, epoch, training_metrics, best=True)
                self.save_best_epoch_visualizations(test_loader, model_path)
            # Save checkpoint every epoch
            self.save_checkpoint(model_path, epoch, training_metrics, best=False)
        # Save final model
        self.save_final(model_path, n_epoch, training_metrics)
    def train_one_epoch(self, train_loader, epoch, training_metrics):
        module = f"{__name__}:train_one_epoch" if self.log_enabled else ""
        self.model.train()
        epoch_loss = 0.0
        epoch_b1_loss = 0.0
        epoch_b2_loss = 0.0
        epoch_combined_loss = 0.0
        epoch_consistency_loss = 0.0
        batch_count = 0
        log_message(f"Starting training for epoch {epoch + 1}", "INFO", module, self.log_enabled, config=[self.config])
        for batch_idx, (_,data,_,target) in enumerate(train_loader):
            try:
                data, target = data.to(self.device), target.to(self.device)
                if batch_idx == 0:
                        log_message(f"Input batch shape: {data.shape}", "INFO", module, self.log_enabled, config=[self.config])
                        log_message(f"Target batch shape: {target.shape}", "INFO", module, self.log_enabled, config=[self.config])

                self.optimizer.zero_grad()
                outputs = self.model(data, target, steps=self.config['nca_steps'])
                if batch_idx == 0:
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                log_message(f"Output {key} shape: {value.shape}", "INFO", module, self.log_enabled, config=[self.config])
                loss, loss_components = self.loss_fn(outputs, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Accumulate metrics
                epoch_loss += loss_components['total_loss']
                if 'b1_loss' in loss_components:
                    epoch_b1_loss += loss_components['b1_loss']
                if 'b2_loss' in loss_components:
                    epoch_b2_loss += loss_components['b2_loss']
                if 'combined_loss' in loss_components:
                    epoch_combined_loss += loss_components['combined_loss']
                if 'consistency_loss' in loss_components:
                    epoch_consistency_loss += loss_components['consistency_loss']
                batch_count += 1

                # Log batch progress
                if batch_idx % 10 == 0 and self.log_enabled:
                    log_msg = f"  Batch {batch_idx}/{len(train_loader)} - Total: {loss_components['total_loss']:.4f}"
                    if 'combined_loss' in loss_components:
                        log_msg += f" Combined: {loss_components['combined_loss']:.4f}"
                    if 'b1_loss' in loss_components and 'b2_loss' in loss_components:
                        log_msg += f" (B1: {loss_components['b1_loss']:.4f}, B2: {loss_components['b2_loss']:.4f})"
                    log_message(log_msg, "INFO", module, self.log_enabled, config=[self.config])

            except Exception as e:
                log_message(f"Error in batch {batch_idx}: {e}", "ERROR", module, self.log_enabled, config=[self.config])
                traceback.print_exc()
                continue

        # Calculate epoch averages
        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            avg_b1_loss = epoch_b1_loss / batch_count if epoch_b1_loss > 0 else 0
            avg_b2_loss = epoch_b2_loss / batch_count if epoch_b2_loss > 0 else 0
            avg_combined_loss = epoch_combined_loss / batch_count if epoch_combined_loss > 0 else 0
            avg_consistency_loss = epoch_consistency_loss / batch_count if epoch_consistency_loss > 0 else 0

            training_metrics['epoch_losses'].append(avg_epoch_loss)
            training_metrics['b1_losses'].append(avg_b1_loss)
            training_metrics['b2_losses'].append(avg_b2_loss)
            training_metrics['combined_losses'].append(avg_combined_loss)
            training_metrics['consistency_losses'].append(avg_consistency_loss)
            training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            log_message(f"Epoch {epoch + 1} completed:", "SUCCESS", module, self.log_enabled, config=[self.config])
            log_message(f"  - Total Loss: {avg_epoch_loss:.4f}", "INFO", module, self.log_enabled, config=[self.config])
            if avg_combined_loss > 0:
                log_message(f"  - Combined Loss: {avg_combined_loss:.4f} (PRIMARY)", "INFO", module, self.log_enabled, config=[self.config])
            if avg_b1_loss > 0:
                log_message(f"  - B1 Loss: {avg_b1_loss:.4f}", "INFO", module, self.log_enabled, config=[self.config])
            if avg_b2_loss > 0:
                log_message(f"  - B2 Loss: {avg_b2_loss:.4f}", "INFO", module, self.log_enabled, config=[self.config])
            if avg_consistency_loss > 0:
                log_message(f"  - Consistency Loss: {avg_consistency_loss:.4f}", "INFO", module, self.log_enabled, config=[self.config])
            log_message(f"  - Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}", "INFO", module, self.log_enabled, config=[self.config])

        self.scheduler.step()

    def evaluate(self, test_loader, max_batches=20):
        module = f"{__name__}:evaluate" if self.log_enabled else ""
        self.model.eval()
        eval_loss = 0.0
        eval_combined_loss = 0.0
        eval_batches = 0

        with torch.no_grad():
            for _,data,_,target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data, target, steps=self.config['nca_steps'])
                loss, loss_components = self.loss_fn(outputs, target)
                eval_loss += loss_components['total_loss']
                if 'combined_loss' in loss_components:
                    eval_combined_loss += loss_components['combined_loss']
                eval_batches += 1
                if eval_batches >= max_batches:
                    break

        avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0
        avg_eval_combined = eval_combined_loss / eval_batches if eval_batches > 0 else 0

        if self.log_enabled:
            log_message(f"  - Evaluation Total Loss: {avg_eval_loss:.4f}", "INFO", module, self.log_enabled, config=[self.config])
            if avg_eval_combined > 0:
                log_message(f"  - Evaluation Combined Loss: {avg_eval_combined:.4f}", "INFO", module, self.log_enabled, config=[self.config])

        return avg_eval_loss, avg_eval_combined

    def save_checkpoint(self, model_path, epoch, training_metrics, best=False):
        save_name = 'best_model.pth' if best else f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': training_metrics
        }, os.path.join(model_path, save_name))
        if self.log_enabled:
            log_message(f"  - Model checkpoint saved: {save_name}", "SUCCESS", __name__, self.log_enabled, config=[self.config])

    def save_final(self, model_path, epoch, training_metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_metrics': training_metrics
        }, os.path.join(model_path, 'final_model.pth'))
        if self.log_enabled:
            log_message(f"  - Final model saved", "SUCCESS", __name__, self.log_enabled, config=[self.config])


    # def normalize_shape(arr):
    #     """Ensure image is in HxW format"""
    #     if isinstance(arr, torch.Tensor):
    #         arr = arr.detach().cpu().numpy()
    #     while arr.ndim > 2:
    #         arr = arr[0]
    #     return arr

    @staticmethod
    def dice_score(pred, target, threshold=None):
        """Compute Dice score with optional thresholding. If threshold is None, use soft Dice."""
        pred = np.squeeze(pred)
        target = np.squeeze(target)

        if target.max() > 1:
            target = target / 255.0  # Normalize if in [0, 255]

        if threshold is not None:
            pred = (pred > threshold).astype(np.uint8)
            target = (target > threshold).astype(np.uint8)
        else:
            # Ensure float32 for soft Dice
            pred = pred.astype(np.float32)
            target = target.astype(np.float32)

        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target)
        dice = (2. * intersection) / (union + 1e-8)
        return dice

    def save_prediction_images(self, data_loader, save_dir, num_samples=10):
        """Save prediction images along with input and ground truth"""
        module = f"{__name__}:save_prediction_images" if self.log_enabled else ""
        outputs_dir = os.path.join(save_dir, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        log_message(f"Saving prediction images to: {outputs_dir}", "INFO", module, self.log_enabled, config=[self.config])
        self.model.eval()
        saved_count = 0

        with torch.no_grad():
            for batch_idx, (_, data,_,  target) in enumerate(data_loader):
                if saved_count >= num_samples:
                    break
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data, target, steps=self.config['nca_steps'])
                batch_size = data.shape[0]
                for i in range(min(batch_size, num_samples - saved_count)):
                    input_img = data[i].cpu().numpy()
                    target_img = target[i].cpu().numpy()
                    pred_images = {}
                    if 'b1_pred' in outputs:
                        b1_pred = torch.sigmoid(outputs['b1_pred'][i]).cpu().numpy()
                        pred_images['b1_pred'] = b1_pred
                    if 'combined_pred' in outputs:
                        combined_pred = torch.sigmoid(outputs['combined_pred'][i]).cpu().numpy()
                        pred_images['combined_pred'] = combined_pred
                    self.save_single_prediction(
                        input_img, target_img, pred_images,
                        os.path.join(outputs_dir, f'prediction_{saved_count+1:03d}.png'),
                        saved_count + 1
                    )
                    saved_count += 1
                    if saved_count >= num_samples:
                        break
        log_message(f"Successfully saved {saved_count} prediction images", "SUCCESS", module, self.log_enabled, config=[self.config])


    @staticmethod
    def save_single_prediction(input_img, target_img, pred_images, save_path, sample_num):
        """Create and save a single prediction visualization with Dice score"""
        def normalize_display(img):
            img = np.squeeze(img)
            if img.max() > 1:
                img = img / 255.0
            return img

        input_display = normalize_display(input_img[0]) if input_img.ndim == 3 else normalize_display(input_img)
        target_display = normalize_display(target_img)

        b1_pred = pred_images.get("b1_pred", np.zeros_like(target_display))
        combined_pred = pred_images.get("combined_pred", np.zeros_like(target_display))

        combined_display = normalize_display(combined_pred)

        # Compute Dice (threshold can be passed or left as None)
        dice = Agent_EnhancedGraphMedNCA.dice_score(combined_display, target_display, threshold=None)

        # Display in 2x2 matrix
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        axes[0, 0].imshow(input_display, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(target_display, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(b1_pred[0] if b1_pred.ndim == 3 else b1_pred, cmap='gray')
        axes[1, 0].set_title('B1 Prediction')
        axes[1, 0].axis('off')

        combined_temp = combined_pred[0] if combined_pred.ndim == 3 else combined_pred
        axes[1, 1].imshow(combined_temp, cmap='gray')
        axes[1, 1].set_title(f'Combined Prediction\nDice: {dice:.4f}')
        axes[1, 1].axis('off')

        plt.suptitle(f'Sample {sample_num}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    def save_best_epoch_visualizations(self, test_loader, model_path):
        """Save visualizations when best model is achieved"""
        module = f"{__name__}:save_best_epoch_visualizations" if self.log_enabled else ""
        log_message("Saving best epoch visualizations...", "INFO", module, self.log_enabled, config=[self.config])
        self.save_prediction_images(test_loader, model_path, num_samples=20)
        log_message("Best epoch visualizations saved", "SUCCESS", module, self.log_enabled, config=[self.config])


    # def save_prediction_images(self, data_loader, save_dir, num_samples=10):
    #     """Save prediction images along with input and ground truth"""
    #     module = f"{__name__}:save_prediction_images" if self.log_enabled else ""
    #     outputs_dir = os.path.join(save_dir, 'outputs')
    #     os.makedirs(outputs_dir, exist_ok=True)
    #     log_message(f"Saving prediction images to: {outputs_dir}", "INFO", module, self.log_enabled, config=[self.config])
    #     self.model.eval()
    #     saved_count = 0

    #     with torch.no_grad():
    #         for batch_idx, (_, _, data, target) in enumerate(data_loader):
    #             if saved_count >= num_samples:
    #                 break
    #             data, target = data.to(self.device), target.to(self.device)
    #             outputs = self.model(data, target, steps=self.config['nca_steps'])
    #             batch_size = data.shape[0]
    #             for i in range(min(batch_size, num_samples - saved_count)):
    #                 input_img = data[i].cpu().numpy()
    #                 target_img = target[i].cpu().numpy()
    #                 pred_images = {}
    #                 if 'b1_pred' in outputs:
    #                     b1_pred = torch.sigmoid(outputs['b1_pred'][i]).cpu().numpy()
    #                     pred_images['b1_pred'] = b1_pred
    #                 if 'b2_pred' in outputs:
    #                     b2_pred = torch.sigmoid(outputs['b2_pred'][i]).cpu().numpy()
    #                     pred_images['b2_pred'] = b2_pred
    #                 if 'combined_pred' in outputs:
    #                     combined_pred = torch.sigmoid(outputs['combined_pred'][i]).cpu().numpy()
    #                     pred_images['combined_pred'] = combined_pred
    #                 self.save_single_prediction(
    #                     input_img, target_img, pred_images,
    #                     os.path.join(outputs_dir, f'prediction_{saved_count+1:03d}.png'),
    #                     saved_count + 1
    #                 )
    #                 saved_count += 1
    #                 if saved_count >= num_samples:
    #                     break
    #     log_message(f"Successfully saved {saved_count} prediction images", "SUCCESS", module, self.log_enabled, config=[self.config])





    # @staticmethod
    # def save_single_prediction(input_img, target_img, pred_images, save_path, sample_num):
    #     """Create and save a single prediction visualization"""
    #     import matplotlib.pyplot as plt
    #     # Normalize input image for display
    #     if input_img.shape[0] == 1:
    #         input_display = input_img[0]
    #     else:
    #         input_display = input_img[0] if input_img.shape[0] > 1 else input_img
    #     if len(target_img.shape) == 3 and target_img.shape[0] == 1:
    #         target_display = target_img[0]
    #     else:
    #         target_display = target_img
    #     num_preds = len(pred_images)
    #     total_cols = 2 + num_preds
    #     fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))
    #     if total_cols == 1:
    #         axes = [axes]
    #     axes[0].imshow(input_display, cmap='gray')
    #     axes[0].set_title('Input Image')
    #     axes[0].axis('off')
    #     axes[1].imshow(target_display, cmap='gray')
    #     axes[1].set_title('Ground Truth')
    #     axes[1].axis('off')
    #     col_idx = 2
    #     for pred_name, pred_data in pred_images.items():
    #         if len(pred_data.shape) == 3 and pred_data.shape[0] == 1:
    #             pred_display = pred_data[0]
    #         else:
    #             pred_display = pred_data
    #         axes[col_idx].imshow(pred_display, cmap='gray')
    #         axes[col_idx].set_title(f'{pred_name.replace("_", " ").title()}')
    #         axes[col_idx].axis('off')
    #         col_idx += 1
    #     plt.suptitle(f'Sample {sample_num}', fontsize=16)
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #     plt.close()