
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# Author: Wentao Yuan (adapted for M2T2 grasp prediction)
# TODO: CHeck all the torch.eye with world coordinates etc
# TODO this is not most efficient as normally it can get input object
# TODO Use ply OR open3D. we use open3D for now, but check if it has drawbacks
# TODO CHECK COLOR OF ITEM

"""
M2T2 Neural Network Grasp Prediction Interface.

This module provides a drop-in replacement for GPD (Grasp Pose Detection) that uses
the M2T2 (Multimodal Manipulation Transformer with Multimodal Tokens) neural network
for robotic grasp prediction. It maintains API compatibility with the original GPD
interface while leveraging the advanced capabilities of transformer-based deep learning.

Key Features:
- Drop-in replacement for GPD with same function signatures
- Neural network-based grasp prediction using M2T2
- Support for point cloud input from various sources (PCL, Open3D)
- Confidence-based grasp ranking and selection

Main Classes:
- M2T2Predictor: Core prediction engine wrapping the M2T2 model
- PointCloud: Point cloud data wrapper for API compatibility
- Config: Configuration parameter container
- Logger: Simple logging utility

Main Functions:
- predict_full_grasp(): High-level interface compatible with Flask server
- _predict(): Lower-level prediction function
- get_predictor(): Singleton accessor for the global predictor instance
"""
import os
import sys
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from m2t2.dataset import collate
from m2t2.dataset_utils import sample_points
from m2t2.m2t2 import M2T2
from m2t2.train_utils import to_cpu, to_gpu

# TODO tf is that check if needed and why it iis assumed
RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
# VISUALIZE = os.environ.get("M2T2_VISUALIZE", "0").lower() in {"1", "true", "yes"}
VISUALIZE = True #True

def _visualize_results(
    scene_points,
    scene_colors,
    object_points,
    object_colors,
    grasps,
    widths,
    scores,
):
    if not VISUALIZE:
        return

    try:
        from m2t2.meshcat_utils import (
            create_visualizer,
            make_frame,
            visualize_grasp,
            visualize_pointcloud,
        )
    except ImportError:
        print("VISUALIZE enabled but meshcat utilities are unavailable; skipping view")
        return

    def _to_uint8(colors):
        colors = colors.detach().cpu().numpy()
        colors = np.clip(colors, 0.0, 1.0)
        return (colors * 255).astype(np.uint8)

    scene_xyz = scene_points.detach().cpu().numpy()
    scene_rgb = _to_uint8(scene_colors)
    obj_xyz = object_points.detach().cpu().numpy()
    obj_rgb = _to_uint8(object_colors)

    vis = create_visualizer()
    make_frame(vis, "world")
    visualize_pointcloud(vis, "scene", scene_xyz, scene_rgb, size=0.003)
    visualize_pointcloud(vis, "object", obj_xyz, obj_rgb, size=0.003)

    if isinstance(grasps, np.ndarray):
        grasp_mats = grasps
    else:
        grasp_mats = np.asarray(grasps)

    for idx, mat in enumerate(grasp_mats):
        name = f"grasp/{idx:03d}"
        visualize_grasp(vis, name, mat, [0, 255, 0], linewidth=0.2)

    print(
        "Meshcat visualizer running; open the displayed URL to inspect grasps"
    )


class PointCloud:
    """Point cloud container that optionally stores per-point RGB values."""

    def __init__(self, points, colors=None):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be an array of shape (N, 3)")
        if points.shape[0] == 0:
            raise ValueError("point cloud cannot be empty")

        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32)
            if colors.shape != points.shape:
                raise ValueError("colors must match points shape (N, 3)")
        self.points = points
        self.colors = colors

    def to_array(self):
        return self.points

    def color_array(self):
        return self.colors


class Config:
    """
    Configuration class for grasp prediction parameters.
    
    This class stores gripper and object parameters that would be used
    by traditional grasp planners. Currently maintained for API compatibility
    but not actively used by the M2T2 neural network.
    
    Attributes:
        gripper_width (float): Maximum gripper opening width in meters.
        finger_depth (float): Depth of gripper fingers in meters.
        hand_depth (float): Depth of the gripper hand in meters.
        object_min_height (float): Minimum object height to consider in meters.
    """
    def __init__(self, **kwargs):
        """Initialize configuration with default values.
        
        Args:
            **kwargs: Keyword arguments for configuration parameters.
        """
        self.gripper_width = kwargs.get('gripper_width', 0.08)
        self.finger_depth = kwargs.get('finger_depth', 0.05)
        self.hand_depth = kwargs.get('hand_depth', 0.10)
        self.object_min_height = kwargs.get('object_min_height', 0.005)


class Logger:
    """
    Simple logging utility for debugging and status messages.
    
    Provides a consistent logging interface compatible with the original
    GPD implementation. Messages are printed to stdout with level indicators.
    
    Attributes:
        name (str): Name identifier for the logger instance.
    """
    def __init__(self, name):
        """Initialize logger with a name.
        
        Args:
            name (str): Name identifier for log messages.
        """
        self.name = name
        
    def info(self, msg):
        """Log an info message.
        
        Args:
            msg (str): Message to log.
        """
        print(f"[{self.name}] INFO: {msg}")
        
    def warn(self, msg):
        """Log a warning message.
        
        Args:
            msg (str): Message to log.
        """
        print(f"[{self.name}] WARN: {msg}")
        
    def error(self, msg):
        """Log an error message.
        
        Args:
            msg (str): Message to log.
        """
        print(f"[{self.name}] ERROR: {msg}")


class M2T2Predictor:
    """
    M2T2 Neural Network Grasp Predictor.
    
    This class wraps the M2T2 (Multimodal Manipulation Transformer with Multimodal Tokens)
    neural network for robotic grasp prediction. It provides a high-level interface
    for loading the model, processing point clouds, and generating grasp poses.
    
    The M2T2 model is a transformer-based architecture that processes point cloud
    data to predict feasible grasp poses with confidence scores.
    
    Attributes:
        cfg (DictConfig): Loaded configuration from config.yaml
        model (M2T2): The neural network model
        device (torch.device): Computing device (CPU or CUDA)
    """
    
    def __init__(self, config_path=None, checkpoint_path=None):
        """
        Initialize the M2T2 predictor with model configuration and trained weights.
        
        Args:
            config_path (str, optional): Path to the configuration YAML file.
                                       Defaults to 'config.yaml' in the current directory.
            checkpoint_path (str, optional): Path to the trained model checkpoint.
                                            Defaults to 'models/m2t2.pth'.
                                            
        Raises:
            FileNotFoundError: If the configuration file is not found.
            RuntimeError: If there are issues loading the model or checkpoint.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'models', 'm2t2.pth')
            
        # Load configuration
        self.cfg = OmegaConf.load(config_path)
        
        # Initialize model
        self.model = M2T2.from_config(self.cfg.m2t2)
        
        # Load checkpoint if available
        if os.path.exists(checkpoint_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(ckpt)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        # USE GPU IF AVAILABLE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"M2T2 model initialized on device: {self.device}")
        
    def _extract_arrays(self, cloud, name):
        if isinstance(cloud, PointCloud):
            points = cloud.to_array()
            colors = cloud.color_array()
        else:
            raise TypeError(f"{name} must be a PointCloud instance")

        if colors is None:
            raise ValueError(f"{name} is missing color information")

        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max() > 1.5:
            colors = colors / 255.0
        return points, colors

    def _build_scene_tensors(self, item_points, item_colors, env_points, env_colors):
        """Concatenate object and environment tensors and build segmentation mask."""

        if env_points.numel() == 0:
            scene_points = item_points.clone()
            scene_colors = item_colors.clone()
        else:
            scene_points = torch.cat([item_points, env_points], dim=0)
            scene_colors = torch.cat([item_colors, env_colors], dim=0)

        seg = torch.zeros(scene_points.shape[0], dtype=torch.bool)
        seg[: item_points.shape[0]] = True

        return scene_points, scene_colors, seg

    def _select_target_object(self, contact_sets, target_points, target_bounds=None):
        """Choose the predicted object whose contacts are closest to the target cloud.
        
        Enhanced version that uses spatial overlap and statistical measures for better matching.
        """

        if not contact_sets:
            return None

        target_points = target_points.to(torch.float32)
        
        # Calculate target object bounds for spatial filtering
        if target_bounds is None:
            target_min = target_points.min(dim=0).values
            target_max = target_points.max(dim=0).values
            # Expand bounds slightly to account for grasp approach margins
            margin = 0.02  # 2cm margin
            target_bounds = (target_min - margin, target_max + margin)
        
        best_idx, best_score = None, float('inf')
        candidate_scores = []

        for idx, contacts in enumerate(contact_sets):
            if not isinstance(contacts, torch.Tensor):
                contacts = torch.as_tensor(contacts)
            contact_points = contacts.to(torch.float32)
            if contact_points.ndim != 2 or contact_points.shape[0] == 0:
                continue
            
            # Filter contacts that are within target object bounds
            target_min, target_max = target_bounds
            within_bounds = (
                (contact_points >= target_min.unsqueeze(0)).all(dim=1) &
                (contact_points <= target_max.unsqueeze(0)).all(dim=1)
            )
            
            if not within_bounds.any():
                # No contacts within target bounds - likely wrong object
                candidate_scores.append(float('inf'))
                continue
            
            # Use only contacts within bounds for distance calculation
            valid_contacts = contact_points[within_bounds]
            
            try:
                dists = torch.cdist(valid_contacts, target_points)
            except RuntimeError:
                # Fallback for extremely large point sets where cdist can OOM
                valid_contacts = valid_contacts[:512]
                target_subset = target_points[:1024]
                dists = torch.cdist(valid_contacts, target_subset)
            
            # Calculate multiple scoring metrics
            min_dists = dists.min(dim=1).values
            mean_dist = min_dists.mean().item()
            median_dist = min_dists.median().item()
            overlap_ratio = within_bounds.float().mean().item()
            
            # Combined score: lower is better
            # Prioritize objects with high overlap and low distances
            score = mean_dist * 0.6 + median_dist * 0.3 + (1.0 - overlap_ratio) * 0.1
            candidate_scores.append(score)
            
            if score < best_score:
                best_score = score
                best_idx = idx
        
        return best_idx

    def predict_from_clouds(self, item_cloud, env_cloud, **kwargs):
        """Run M2T2 inference on explicit object/environment point clouds."""

        item_points_np, item_colors_np = self._extract_arrays(item_cloud, 'item_cloud')
        env_points_np, env_colors_np = self._extract_arrays(env_cloud, 'env_cloud')

        item_points = torch.from_numpy(item_points_np)
        env_points = torch.from_numpy(env_points_np)
        item_colors = torch.from_numpy(item_colors_np)
        env_colors = torch.from_numpy(env_colors_np)

        scene_points, scene_colors, seg = self._build_scene_tensors(
            item_points, item_colors, env_points, env_colors
        )

        # Store original target object bounds before sampling
        target_bounds = (
            item_points.min(dim=0).values,
            item_points.max(dim=0).values
        )
        
        scene_idx = sample_points(scene_points, self.cfg.data.num_points)
        scene_points = scene_points[scene_idx]
        scene_colors = scene_colors[scene_idx]
        seg = seg[scene_idx]

        scene_points_vis = scene_points.clone()
        scene_colors_vis = scene_colors.clone()

        centered_scene = scene_points - scene_points.mean(dim=0, keepdim=True)
        rgb_mean = RGB_MEAN.to(centered_scene.device)
        rgb_std = RGB_STD.to(centered_scene.device)
        normalized_scene_colors = (scene_colors - rgb_mean) / rgb_std
        inputs = torch.cat([centered_scene, normalized_scene_colors], dim=1)

        object_idx = sample_points(item_points, self.cfg.data.num_object_points)
        obj_points = item_points[object_idx]
        obj_colors = item_colors[object_idx]
        obj_points_vis = obj_points.clone()
        obj_colors_vis = obj_colors.clone()
        centered_obj = obj_points - obj_points.mean(dim=0, keepdim=True)
        normalized_obj_colors = (obj_colors - rgb_mean) / rgb_std
        object_inputs = torch.cat([centered_obj, normalized_obj_colors], dim=1)

        object_center = item_points.mean(dim=0)
        bottom_center = item_points.min(dim=0).values

        placement_masks = torch.zeros(
            self.cfg.data.num_rotations, inputs.shape[0]
        )
        placement_region = torch.zeros(inputs.shape[0])

        data = {
            'inputs': inputs,
            'points': scene_points,
            'seg': seg,
            'object_inputs': object_inputs,
            'task': 'pick',
            'cam_pose': torch.eye(4),
            'ee_pose': torch.eye(4),
            'bottom_center': bottom_center,
            'object_center': object_center,
            'placement_masks': placement_masks,
            'placement_region': placement_region,
        }

        batch = collate([data])

        if self.device.type == 'cuda':
            to_gpu(batch)

        with torch.no_grad():
            outputs = self.model.infer(batch, self.cfg.eval)
        

        to_cpu(outputs)

        tf_matrices, widths, scores = [], [], []
        n_best = kwargs.get('n_best', 1)

        grasps_batches = outputs.get('grasps', [])
        confidences_batches = outputs.get('grasp_confidence', [])
        contacts_batches = outputs.get('grasp_contacts', [])
        objectness_batches = outputs.get('objectness')

        for batch_idx, grasp_objects in enumerate(grasps_batches):
            if not grasp_objects:
                continue

            conf_objects = (
                confidences_batches[batch_idx]
                if batch_idx < len(confidences_batches)
                else []
            )
            contact_objects = (
                contacts_batches[batch_idx]
                if batch_idx < len(contacts_batches)
                else []
            )
            objectness_objects = None
            if objectness_batches is not None and batch_idx < len(objectness_batches):
                objectness_objects = objectness_batches[batch_idx]

            target_idx = self._select_target_object(contact_objects, obj_points, target_bounds)
            if target_idx is None or target_idx >= len(grasp_objects):
                print(f"Warning: Could not find valid target object. Available objects: {len(grasp_objects)}")
                continue

            grasp_tensor = grasp_objects[target_idx]
            if grasp_tensor.numel() == 0:
                continue

            # Additional validation: check if grasps are near target object
            if target_idx < len(contact_objects) and len(contact_objects[target_idx]) > 0:
                contacts = contact_objects[target_idx]
                if isinstance(contacts, torch.Tensor):
                    contact_points = contacts.to(torch.float32)
                    target_min, target_max = target_bounds
                    # Check if most contacts are within expanded target bounds
                    margin = 0.05  # 5cm margin for grasp approaches
                    expanded_min = target_min - margin
                    expanded_max = target_max + margin
                    within_expanded = (
                        (contact_points >= expanded_min.unsqueeze(0)).all(dim=1) &
                        (contact_points <= expanded_max.unsqueeze(0)).all(dim=1)
                    )
                    overlap_ratio = within_expanded.float().mean().item()
                    if overlap_ratio < 0.3:  # Less than 30% overlap - likely wrong object
                        print(f"Warning: Selected object {target_idx} has low spatial overlap ({overlap_ratio:.2f}) with target")
                        continue

            if target_idx < len(conf_objects):
                conf_tensor = conf_objects[target_idx]
            else:
                conf_tensor = torch.ones(grasp_tensor.shape[0])

            if objectness_objects is not None and target_idx < len(objectness_objects):
                obj_val = objectness_objects[target_idx]
                object_score = (
                    float(obj_val)
                    if not isinstance(obj_val, torch.Tensor)
                    else float(obj_val.item())
                )
                if object_score <= self.cfg.eval.object_thresh:
                    continue

            grasp_np = grasp_tensor.numpy()
            conf_np = (
                conf_tensor.numpy()
                if isinstance(conf_tensor, torch.Tensor)
                else np.asarray(conf_tensor)
            )
            if conf_np.ndim == 0:
                conf_np = np.full(grasp_np.shape[0], float(conf_np))

            top_idx = np.argsort(conf_np)[-n_best:][::-1]
            for idx in top_idx:
                tf_matrices.append(grasp_np[idx])
                widths.append(0.08)
                scores.append(conf_np[idx])

        if tf_matrices:
            tf_arr = np.asarray(tf_matrices)
            widths_arr = np.asarray(widths)
            scores_arr = np.asarray(scores)
        else:
            tf_arr = np.empty((0, 4, 4), dtype=np.float32)
            widths_arr = np.empty(0, dtype=np.float32)
            scores_arr = np.empty(0, dtype=np.float32)

        _visualize_results(
            scene_points_vis,
            scene_colors_vis,
            obj_points_vis,
            obj_colors_vis,
            tf_arr,
            widths_arr,
            scores_arr,
        )

        return tf_arr, widths_arr, scores_arr


# Global predictor instance
_predictor = None


def get_predictor():
    """
    Get or create the global M2T2 predictor instance.
    
    This function implements a singleton pattern to ensure only one model
    instance is loaded in memory, which is more efficient for repeated
    predictions.
    
    Args:
    Returns:
        M2T2Predictor: The global predictor instance.
        
    Note:
        The predictor is initialized lazily on first call.
    """
    global _predictor
    if _predictor is None:
        _predictor = M2T2Predictor()
    return _predictor


def _predict(
    item_cloud,
    env_cloud,
    config=None,
    logger=None,
    top_n=3,
    n_best=1,
):
    """
    Predict grasp poses using M2T2 neural network.
    
    This function serves as the main interface for grasp prediction, maintaining
    compatibility with the original GPD API while using M2T2 internally.
    
    Args:
        item_cloud (PointCloud or np.ndarray): Point cloud of the object to grasp
        env_cloud (PointCloud or np.ndarray): Point cloud of the environment
        config (Config, optional): Configuration parameters (currently unused)
        logger (Logger, optional): Logger instance for debugging
        top_n (int): Number of grasps per rotation (passed to M2T2)
        n_best (int): Number of best grasps to return
        
    Returns:
        tuple: (tf_matrices, widths, scores)
            - tf_matrices (np.ndarray): Array of 4x4 transformation matrices
            - widths (np.ndarray): Array of gripper widths
            - scores (np.ndarray): Array of confidence scores
    """
    if logger is None:
        logger = Logger("M2T2Predictor")
        
    try:
        predictor = get_predictor()
        return predictor.predict_from_clouds(
            item_cloud, env_cloud,
            top_n=top_n, n_best=n_best
        )
    except Exception as e:
        logger.error(f"M2T2 prediction failed: {str(e)}")
        logger.error("Returning empty results. Check model checkpoint and input data.")
        # Return empty arrays with correct shapes to maintain API compatibility
        return np.empty((0, 4, 4)), np.empty(0), np.empty(0)


def predict_full_grasp(
    item_cloud,
    env_cloud,
    config=None,
    logger=None,
    rotation_resolution=24,  # Kept for API compatibility but unused
    top_n=3,
    n_best=1,
    vis_block=False,  # Kept for API compatibility but unused
):
    """
    Full grasp prediction function compatible with the Flask server API.
    
    This function maintains the exact same signature as the original GPD
    implementation to ensure compatibility with app_server.py, but internally
    uses the M2T2 neural network for prediction.
    
    Args:
        item_cloud (PointCloud): Point cloud of the object to grasp
        env_cloud (PointCloud): Point cloud of the environment
        config (Config, optional): Configuration parameters (currently unused)
        logger (Logger, optional): Logger instance for debugging
        rotation_resolution (int): Number of rotations (kept for compatibility, unused)
        top_n (int): Number of grasps per rotation
        n_best (int): Number of best grasps to return
        vis_block (bool): Visualization flag (kept for compatibility, unused)
        
    Returns:
        tuple: (tf_matrices, widths, scores)
            - tf_matrices (np.ndarray): Array of 4x4 transformation matrices
            - widths (np.ndarray): Array of gripper widths  
            - scores (np.ndarray): Array of confidence scores
    """
    return _predict(
        item_cloud=item_cloud,
        env_cloud=env_cloud,
        config=config,
        logger=logger,
        top_n=top_n,
        n_best=n_best
    )
