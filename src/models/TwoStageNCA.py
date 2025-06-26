import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageNCA(nn.Module):
    """Two-stage NCA model that combines GraphMedNCA for low-res segmentation
    followed by BasicNCA for high-res refinement"""
    def __init__(self, stage1_model, stage2_model, stage1_size=(64, 64), stage2_size=(256, 256),stage1_steps=1, stage2_steps=1, 
                 device='cuda', log_enabled=True):
        super(TwoStageNCA, self).__init__()
        self.stage1_model = stage1_model  
        self.stage2_model = stage2_model 
        self.nca_steps_stage1 = stage1_steps
        self.nca_steps_stage2 = stage2_steps 
        self.stage1_size = stage1_size
        self.stage2_size = stage2_size
        self.device = device
        self.log_enabled = log_enabled
    
    def parameters(self, recurse = True):
        return super().parameters(recurse)

    def forward(self, x, steps: int|list[int]=1, return_graph=None):
        if return_graph is None:
            return_graph = (False, False)
        # Downsample input for stage 1
        x_small = F.interpolate(x, size=self.stage1_size, mode='bilinear', align_corners=False)

        # stage_1_steps = steps if isinstance(steps, int) else steps[0]
        stage_1_steps = self.nca_steps_stage1
        stage_2_steps = self.nca_steps_stage2

        # stage_2_steps = steps if isinstance(steps, int) else steps[0]
        edge_idx1 = None
        edge_idx2 = None
        _im1 = None
        _im2 = None
        
        # Stage 1 for rough segmentation on low-res
        if return_graph[0]:
            stage1_output, (_im1, edge_idx1) = self.stage1_model(x_small, steps=stage_1_steps, return_graph=True)
        else:
            stage1_output = self.stage1_model(x_small, steps=stage_1_steps)
        
        # Upsample stage1 output to stage2 size
        stage1_upsampled = F.interpolate(stage1_output, size=self.stage2_size, mode='bilinear', align_corners=False)
        
        # Stage 2 for refinement - following the figure, we use the upsampled segmentation
        # from stage 1 as the input to stage 2, not concatenated with original image
        if return_graph[1]:
            stage2_output, (_im2, edge_idx2) = self.stage2_model(stage1_upsampled, steps=stage_2_steps, return_graph=True)
        else:
            stage2_output = self.stage2_model(stage1_upsampled, steps=stage_2_steps)

        
        return {
            'stage1_output': stage1_output,
            'stage1_upsampled': stage1_upsampled,
            'stage2_output': stage2_output
        }, [(_im1, edge_idx1), (_im2, edge_idx2)] 
    
    def get_final_output(self, x):
        """Returns only the final output of the two-stage model"""
        return self.forward(x)['stage2_output']