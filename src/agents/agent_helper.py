import torch
import os
from tqdm import tqdm
from src.utils.helper import log_message
import matplotlib.pyplot as plt
import numpy as np

def dice_coefficient(pred, target):
    smooth = 1e-7
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def evaluate_imsave(agent, output_dir=None, **kwargs):
    module = f"{__name__}:evaluate_imsave" if agent.log_enabled else ""
    
    if agent.experiment is None:
        log_message("Experiment not set. Call set_exp() before evaluation.", "ERROR", module, agent.log_enabled, config=agent.projectConfig)
        return 0.0
        
    if output_dir is None:
        output_dir = os.path.join(agent.projectConfig[0]['model_path'], "outputs")
    os.makedirs(output_dir, exist_ok=True)
    log_message(f"Evaluating model and saving outputs to {output_dir}", "INFO", module, agent.log_enabled, config=agent.projectConfig)
    agent.model.eval()
    device_str = str(agent.model.device)
    log_message(f"Model is using device: {device_str}", "INFO", module, agent.log_enabled, config=agent.projectConfig)
    try:
        if agent.experiment.test_dataset is None:
            dataset = agent.experiment.dataset
            
            try:
                test_ratio = agent.experiment.get_from_config('data_split', [0.7, 0, 0.3])[2]
                if test_ratio <= 0 or test_ratio >= 1:
                    log_message(f"Invalid {test_ratio=}, using default last 30% of dataset", "WARNING", module, agent.log_enabled, config=agent.projectConfig)
                    test_ratio = 0.3
                test_indices = list(range(len(dataset)))[-int(len(dataset) * test_ratio):]
                log_message("Using fallback test indices (last 30% of dataset)", "WARNING", module, agent.log_enabled, config=agent.projectConfig)
            except Exception as inner_e:
                log_message(f"Could not determine test indices: {str(inner_e)}", "ERROR", module, agent.log_enabled, config=agent.projectConfig)
                test_indices = []
        
        else:
            dataset = agent.experiment.test_dataset
            test_indices = list(range(len(dataset)))
            log_message("Using all indices from test dataset", "INFO", module, agent.log_enabled, config=agent.projectConfig)
        
        if not test_indices:
            log_message("No test data available for evaluation", "WARNING", module, agent.log_enabled, config=agent.projectConfig)
            return 0.0
            
        log_message(f"Evaluating model on {len(test_indices)} test images", "INFO", module, agent.log_enabled, config=agent.projectConfig)
        steps = agent.experiment.get_from_config('inference_steps')
        if steps is None:
            steps = 10  
            log_message(f"No inference_steps in config, using default: {steps}", "WARNING", module, agent.log_enabled, config=agent.projectConfig)
        # Inference Loop
        total_dice = 0.0
        with torch.no_grad():
            for idx in tqdm(test_indices):
                item = dataset[idx]
                if isinstance(item, tuple) and len(item) == 4:
                    img_id, original_img, img, label = item
                else:
                    log_message(f"Invalid item format at index {idx}: {item}", "ERROR", module, agent.log_enabled, config=agent.projectConfig)
                    continue
                    
                # Ensure proper tensor dimensions [batch, channel, height, width] -> adds batch size = 1
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                if len(label.shape) == 3: 
                    label = label.unsqueeze(0)
                img = img.to(agent.model.device)
                label = label.to(agent.model.device)
                graph_returns = kwargs.get('graph_returns', None)
                if graph_returns:
                    prediction = agent.model(img, steps=steps, return_graph=graph_returns)
                else:
                    prediction = agent.model(img, steps=steps)
                
                edge_index = []
                im_tensors = []
                
                if isinstance(prediction, tuple):
                    # (pred, [(im, edge), ...])
                    prediction, edge_index_pairs = prediction
                
                    if isinstance(edge_index_pairs, list):
                        for _im, _edge in edge_index_pairs:
                            edge_index.append(_edge)
                            im_tensors.append(_im)
                    elif isinstance(edge_index_pairs, tuple):
                        edge_index = [edge_index_pairs[1]]
                        im_tensors = [edge_index_pairs[0]]

                if isinstance(prediction, dict):
                    stage1_output = prediction.get('stage1_upsampled')
                    stage2_output = prediction.get('stage2_output')
                    if stage2_output is None or stage1_output is None:
                        log_message(f"Missing outputs in prediction for index {idx}: {prediction}", "ERROR", module, agent.log_enabled, config=agent.projectConfig)
                        continue
                else:
                    stage1_output = None
                    stage2_output = prediction
                
                # if 'threshold' not in kwargs:
                #     stage1_pred = stage1_output if stage1_output is not None else None
                #     stage2_pred = stage2_output
                # else:
                #     stage1_pred = (stage1_output > kwargs['threshold']).float() if stage1_output is not None else None
                #     stage2_pred = (stage2_output > kwargs['threshold']).float()
                # APPLY THRESHOLD LATER (imsave function)
                stage1_pred = stage1_output if stage1_output is not None else None
                stage2_pred = stage2_output
                
                _dice = dice_coefficient(stage2_output, label)
                total_dice += _dice

                # if idx % 10 == 0:
                #     log_message(f"Processed index {idx}, Dice score: {_dice:.4f}", "INFO", module, agent.log_enabled, config=agent.projectConfig)
                
                if 'images_to_save' in kwargs and isinstance(kwargs['images_to_save'], int):
                    if idx > kwargs['images_to_save']:
                        # log_message(f"Reached maximum images to save: {kwargs['images_to_save']}", "INFO", module, agent.log_enabled, config=agent.projectConfig)
                        continue

                if isinstance(img_id, str):
                    dir_name = ''.join(c for c in img_id if c.isalnum() or c in '._-')
                else:
                    dir_name = f"image_{idx}"
                save_path = os.path.join(output_dir, f"{dir_name}_{_dice:.2f}")
                imsave(
                    path=save_path,
                    original_img=original_img.cpu().numpy(),
                    ground_truth_image=label[0].cpu().numpy(),
                    images_to_save=[
                        stage1_pred[0].cpu().numpy() if stage1_pred is not None else None,
                        stage2_pred[0].cpu().numpy()
                    ],
                    threshold=kwargs.get('threshold', None)
                )
                if kwargs.get('save_graphs', False):
                    graph_save(
                        path=save_path,
                        edge_index=edge_index,
                        im_tensors=im_tensors
                    )
        
        avg_dice = total_dice / len(test_indices) if test_indices else 0.0
        log_message(f"Evaluation completed. Average Dice Score: {avg_dice:.4f}", "SUCCESS", module, agent.log_enabled, config=agent.projectConfig)
        log_message(f"Segmentation maps saved to {output_dir}", "SUCCESS", module, agent.log_enabled, config=agent.projectConfig)
        return avg_dice
    except Exception as e:
        log_message(f"Error during evaluation: {str(e)}", "ERROR", module, agent.log_enabled, config=agent.projectConfig)
        import traceback
        traceback.print_exc()
        return 0.0


def graph_save(path, edge_index, im_tensors):
    os.makedirs(path, exist_ok=True)
    # save graph as data file
    for i, (_edge, _im) in enumerate(zip(edge_index, im_tensors)):
        if _edge is None or _im is None:
            continue

        torch.save(_edge, os.path.join(path, f'graph_{i}.pt'))
        torch.save(_im, os.path.join(path, f'image_tensor_{i}.pt'))
        # edges = edges.cpu().numpy() if isinstance(edges, torch.Tensor) else edges
        # np.savetxt(os.path.join(path, f'graph_{i}.txt'), edges, fmt='%d', delimiter=',')
    

def imsave(path, original_img, ground_truth_image, images_to_save, threshold=None):
    os.makedirs(path, exist_ok=True)
    if original_img.shape[0] == 3:
        original_img = np.transpose(original_img, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
    else:
        original_img = original_img[0]
    
    if 1.0 < np.max(original_img) <= 255:
        original_img = original_img / 255.0
    plt.imsave(os.path.join(path, 'original.png'), original_img)
    
    # Save ground truth -> batch size = 1 so take first index only
    plt.imsave(os.path.join(path, 'ground_truth.png'), ground_truth_image[0], cmap='gray')
    
    # Save predictions
    for i, img in enumerate(images_to_save):
        if img is not None:
            img_display = img.copy()
            if len(img_display.shape) == 3 and img_display.shape[0] > 0:
                img_display = img_display[0]  # Take first channel if multiple
            plt.imsave(os.path.join(path, f'prediction_{i+1}.png'), img_display, cmap='gray')
            if threshold is not None:
                img_threshold = (img_display > threshold).astype(np.float32)
                plt.imsave(os.path.join(path, f'prediction_{i+1}_threshold.png'), img_threshold, cmap='gray')
    
    # log_message(f"Saved images to {path}", "INFO", agent.module_name, agent.log_enabled, config=agent.projectConfig)