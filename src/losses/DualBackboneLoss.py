import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.LossFunctions import DiceBCELoss


class DualBackboneLoss(nn.Module):
    """Enhanced loss function for dual backbone architecture with proper gradient flow"""
    def __init__(self, b1_weight=0.3, b2_weight=0.2, combined_weight=0.5):
        super(DualBackboneLoss, self).__init__()
        self.b1_weight = b1_weight
        self.b2_weight = b2_weight
        self.combined_weight = combined_weight  # Main loss for integrated output
        self.bce_loss = DiceBCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, target):
        """
        Fixed forward method that properly handles shape mismatches
        """
        total_loss = 0.0
        loss_components = {}
        
        # Ensure target is properly shaped
        if len(target.shape) == 3:
            target = target.unsqueeze(1)  # Add channel dimension if missing
        
        target_size = target.shape[-2:]  # Get spatial dimensions
        batch_size = target.shape[0]
        
        # Primary loss: Combined backbone output (follows orange line in diagram)
        if 'combined_pred' in outputs:
            combined_pred = outputs['combined_pred']
            
            # Fix shape mismatch by upsampling if needed
            if combined_pred.shape[-2:] != target_size:
                combined_pred = F.interpolate(combined_pred, size=target_size, mode='bilinear', align_corners=False)
            
            # Fix batch size mismatch if needed
            if combined_pred.shape[0] != batch_size:
                # If we have more outputs than inputs (due to patching), take the first batch_size
                combined_pred = combined_pred[:batch_size]
            
            combined_loss = self.bce_loss(combined_pred, target)
            total_loss += self.combined_weight * combined_loss
            loss_components['combined_loss'] = combined_loss.item()
        
        # Auxiliary loss: Backbone 1 (256x256 output) - helps training stability
        if 'b1_pred' in outputs:
            b1_pred = outputs['b1_pred']
            
            # Fix shape mismatch
            if b1_pred.shape[-2:] != target_size:
                b1_pred = F.interpolate(b1_pred, size=target_size, mode='bilinear', align_corners=False)
            
            # Fix batch size mismatch
            if b1_pred.shape[0] != batch_size:
                b1_pred = b1_pred[:batch_size]
            
            b1_loss = self.bce_loss(b1_pred, target)
            total_loss += self.b1_weight * b1_loss
            loss_components['b1_loss'] = b1_loss.item()
        
        # Auxiliary loss: Backbone 2 (patch outputs) - helps training stability
        if 'b2_pred' in outputs:
            b2_pred = outputs['b2_pred']
            
            # Fix shape mismatch
            if b2_pred.shape[-2:] != target_size:
                b2_pred = F.interpolate(b2_pred, size=target_size, mode='bilinear', align_corners=False)
            
            # Fix batch size mismatch
            if b2_pred.shape[0] != batch_size:
                b2_pred = b2_pred[:batch_size]
            
            b2_loss = self.bce_loss(b2_pred, target)
            total_loss += self.b2_weight * b2_loss
            loss_components['b2_loss'] = b2_loss.item()
        
        # Feature consistency loss (ensures proper integration)
        if 'b1_features' in outputs and 'b2_features' in outputs and 'combined_features' in outputs:
            # Ensure feature dimensions match for consistency check
            b1_feats = outputs['b1_features']
            b2_feats = outputs['b2_features'] 
            combined_feats = outputs['combined_features']
            
            # Ensure batch sizes match
            min_batch_size = min(b1_feats.shape[0], b2_feats.shape[0], combined_feats.shape[0], batch_size)
            b1_feats = b1_feats[:min_batch_size]
            b2_feats = b2_feats[:min_batch_size]
            combined_feats = combined_feats[:min_batch_size]
            
            # Feature alignment loss - encourages meaningful feature integration
            if b1_feats.shape[-2:] == combined_feats.shape[-2:]:
                # Resize b2_feats to match b1_feats if needed
                if b2_feats.shape[-2:] != b1_feats.shape[-2:]:
                    b2_feats_resized = F.adaptive_avg_pool2d(b2_feats, b1_feats.shape[-2:])
                else:
                    b2_feats_resized = b2_feats
                weighted_feats = torch.cat([
                    0.7 * b1_feats,
                    0.3 * b2_feats_resized
                ], dim=1)
                consistency_loss = self.mse_loss(combined_feats,weighted_feats)
                total_loss += 0.1 * consistency_loss
                loss_components['consistency_loss'] = consistency_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        return total_loss, loss_components