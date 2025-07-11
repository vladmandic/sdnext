import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class OperationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class OperationNode:
    id: str
    name: str
    status: OperationStatus
    parent: Optional[str] = None  # Parent operation ID for hierarchical tracking
    children: List[str] = None  # Child operation IDs
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    details: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.children is None:
            self.children = []


class PipelineVisualizer:
    def __init__(self):
        self.nodes: Dict[str, OperationNode] = {}
        self.current_pipeline_id: Optional[str] = None
        self.enabled = True
        self.operation_stack: List[str] = []  # Stack to track parent operations
        
    def start_pipeline(self, pipeline_id: str):
        """Start tracking a new pipeline"""
        self.current_pipeline_id = pipeline_id
        self.nodes.clear()
        self.operation_stack.clear()
        
    def push_parent_operation(self, parent_id: str):
        """Push a parent operation onto the stack"""
        if parent_id not in self.operation_stack:
            self.operation_stack.append(parent_id)
        
    def pop_parent_operation(self):
        """Pop the most recent parent operation from the stack"""
        if self.operation_stack:
            return self.operation_stack.pop()
        return None
        
    def get_current_parent(self) -> Optional[str]:
        """Get the current parent operation"""
        return self.operation_stack[-1] if self.operation_stack else None
        
    def add_operation(self, op_id: str, name: str, dependencies: List[str] = None, parent: str = None):
        """Add an operation to track"""
        if not self.enabled:
            return
            
        # Auto-detect parent if not specified
        if parent is None:
            parent = self.get_current_parent()
            
        node = OperationNode(
            id=op_id,
            name=name,
            status=OperationStatus.PENDING,
            parent=parent,
            dependencies=dependencies or []
        )
        self.nodes[op_id] = node
        
        # Add this operation as a child of its parent
        if parent and parent in self.nodes:
            self.nodes[parent].children.append(op_id)
            
    def start_operation(self, op_id: str, details: Dict[str, Any] = None):
        """Mark an operation as started"""
        if not self.enabled or op_id not in self.nodes:
            return
            
        node = self.nodes[op_id]
        node.status = OperationStatus.RUNNING
        node.start_time = time.time()
        if details:
            node.details.update(details)
            
    def complete_operation(self, op_id: str, details: Dict[str, Any] = None):
        """Mark an operation as completed"""
        if not self.enabled or op_id not in self.nodes:
            return
            
        node = self.nodes[op_id]
        node.status = OperationStatus.COMPLETED
        node.end_time = time.time()
        if node.start_time:
            node.duration = node.end_time - node.start_time
        if details:
            node.details.update(details)
            
    def fail_operation(self, op_id: str, error: str):
        """Mark an operation as failed"""
        if not self.enabled or op_id not in self.nodes:
            return
            
        node = self.nodes[op_id]
        node.status = OperationStatus.FAILED
        node.end_time = time.time()
        if node.start_time:
            node.duration = node.end_time - node.start_time
        node.details['error'] = error
        
    def skip_operation(self, op_id: str, reason: str = ""):
        """Mark an operation as skipped"""
        if not self.enabled or op_id not in self.nodes:
            return
            
        node = self.nodes[op_id]
        node.status = OperationStatus.SKIPPED
        if reason:
            node.details['skip_reason'] = reason
            
    def get_pipeline_state(self) -> Dict:
        """Get current pipeline state for visualization"""
        return {
            'pipeline_id': self.current_pipeline_id,
            'nodes': [asdict(node) for node in self.nodes.values()],
            'edges': self._get_edges(),
            'hierarchy': self._get_hierarchy(),
            'total_duration': self._get_total_duration(),
            'status': self._get_overall_status()
        }
        
    def _get_edges(self) -> List[Dict]:
        """Generate edges based on dependencies and hierarchy"""
        edges = []
        
        # Dependency edges
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in self.nodes:
                    edges.append({
                        'from': dep,
                        'to': node.id,
                        'type': 'dependency',
                        'status': self._get_edge_status(dep, node.id)
                    })
        
        # Hierarchy edges (parent-child relationships)
        for node in self.nodes.values():
            if node.parent and node.parent in self.nodes:
                edges.append({
                    'from': node.parent,
                    'to': node.id,
                    'type': 'hierarchy',
                    'status': self._get_edge_status(node.parent, node.id)
                })
        
        return edges
        
    def _get_hierarchy(self) -> Dict:
        """Generate hierarchy structure for visualization"""
        hierarchy = {}
        
        # Find root nodes (no parents)
        for node in self.nodes.values():
            if not node.parent:
                hierarchy[node.id] = {
                    'name': node.name,
                    'status': node.status.value,
                    'children': self._get_children_recursive(node.id)
                }
        
        return hierarchy
        
    def _get_children_recursive(self, node_id: str) -> Dict:
        """Recursively get children for hierarchy structure"""
        children = {}
        node = self.nodes.get(node_id)
        
        if node:
            for child_id in node.children:
                if child_id in self.nodes:
                    child_node = self.nodes[child_id]
                    children[child_id] = {
                        'name': child_node.name,
                        'status': child_node.status.value,
                        'children': self._get_children_recursive(child_id)
                    }
        
        return children
        
    def _get_edge_status(self, from_id: str, to_id: str) -> str:
        """Get edge status based on connected nodes"""
        from_node = self.nodes.get(from_id)
        to_node = self.nodes.get(to_id)
        
        if not from_node or not to_node:
            return "inactive"
            
        if from_node.status == OperationStatus.COMPLETED:
            if to_node.status in [OperationStatus.RUNNING, OperationStatus.COMPLETED]:
                return "active"
            elif to_node.status == OperationStatus.FAILED:
                return "error"
        
        return "inactive"
        
    def _get_total_duration(self) -> float:
        """Calculate total pipeline duration"""
        start_times = [node.start_time for node in self.nodes.values() if node.start_time]
        end_times = [node.end_time for node in self.nodes.values() if node.end_time]
        
        if not start_times or not end_times:
            return 0
            
        return max(end_times) - min(start_times)
        
    def _get_overall_status(self) -> str:
        """Get overall pipeline status"""
        statuses = [node.status for node in self.nodes.values()]
        
        if OperationStatus.FAILED in statuses:
            return "failed"
        elif OperationStatus.RUNNING in statuses:
            return "running"
        elif all(s in [OperationStatus.COMPLETED, OperationStatus.SKIPPED] for s in statuses):
            return "completed"
        else:
            return "pending"


# Global instance
pipeline_viz = PipelineVisualizer()


def get_operation_details(op_name: str, p) -> Dict[str, Any]:
    """Get operation-specific details for visualization"""
    details = {}
    
    # Model loading operations
    if op_name == 'model_load_text':
        details.update({
            'text_encoder_type': getattr(p, 'text_encoder_type', 'clip'),
            'clip_skip': getattr(p, 'clip_skip', 1),
            'model_type': getattr(p, 'model_type', 'sd15')
        })
    elif op_name == 'model_load_unet':
        details.update({
            'unet_dtype': getattr(p, 'unet_dtype', 'float16'),
            'unet_device': getattr(p, 'unet_device', 'cuda'),
            'model_type': getattr(p, 'model_type', 'sd15')
        })
    elif op_name == 'model_load_vae':
        details.update({
            'vae_type': getattr(p, 'vae_type', 'standard'),
            'vae_dtype': getattr(p, 'vae_dtype', 'float16'),
            'vae_device': getattr(p, 'vae_device', 'cuda')
        })
    elif op_name == 'model_move_gpu':
        details.update({
            'gpu_id': getattr(p, 'gpu_id', 0),
            'memory_fraction': getattr(p, 'memory_fraction', 0.95)
        })
    elif op_name == 'model_move_cpu':
        details.update({
            'offload_reason': 'memory_optimization'
        })
    elif op_name == 'model_offload':
        details.update({
            'offload_type': getattr(p, 'offload_type', 'cpu'),
            'memory_threshold': getattr(p, 'memory_threshold', 0.8)
        })
    
    # Memory management operations
    elif op_name == 'memory_cleanup':
        details.update({
            'cleanup_type': 'intermediate',
            'gc_collect': True
        })
    elif op_name == 'memory_cleanup_final':
        details.update({
            'cleanup_type': 'final',
            'gc_collect': True,
            'torch_empty_cache': True
        })
    
    # Extension loading operations
    elif op_name == 'lora_load':
        details.update({
            'lora_modules': getattr(p, 'lora_modules', []),
            'lora_weights': getattr(p, 'lora_weights', [])
        })
    elif op_name == 'lora_apply':
        details.update({
            'lora_scale': getattr(p, 'lora_scale', 1.0),
            'lora_merge_mode': getattr(p, 'lora_merge_mode', 'weighted')
        })
    elif op_name == 'embedding_load':
        details.update({
            'embedding_names': getattr(p, 'embedding_names', []),
            'embedding_dir': getattr(p, 'embedding_dir', '')
        })
    elif op_name == 'ipadapter_apply':
        details.update({
            'ipadapter_model': getattr(p, 'ipadapter_model', ''),
            'ipadapter_weight': getattr(p, 'ipadapter_weight', 1.0)
        })
    
    # Scheduler and optimization operations
    elif op_name == 'scheduler_init':
        details.update({
            'scheduler_type': getattr(p, 'sampler_name', ''),
            'num_inference_steps': getattr(p, 'steps', 20),
            'guidance_scale': getattr(p, 'cfg_scale', 7.5)
        })
    elif op_name == 'attention_slicing':
        details.update({
            'slice_size': getattr(p, 'attention_slice_size', 'auto'),
            'memory_efficient': getattr(p, 'memory_efficient_attention', True)
        })
    elif op_name == 'attention_chunking':
        details.update({
            'chunk_size': getattr(p, 'attention_chunk_size', 8),
            'chunk_threshold': getattr(p, 'attention_chunk_threshold', 2048)
        })
    elif op_name == 'gradient_checkpoint':
        details.update({
            'gradient_checkpointing': getattr(p, 'gradient_checkpointing', True),
            'checkpoint_segments': getattr(p, 'checkpoint_segments', 4)
        })
    elif op_name == 'mixed_precision':
        details.update({
            'mixed_precision_policy': getattr(p, 'mixed_precision_policy', 'fp16'),
            'autocast_enabled': getattr(p, 'autocast_enabled', True)
        })
    
    # Noise and scheduling operations
    elif op_name == 'noise_schedule':
        details.update({
            'beta_start': getattr(p, 'beta_start', 0.00085),
            'beta_end': getattr(p, 'beta_end', 0.012),
            'beta_schedule': getattr(p, 'beta_schedule', 'scaled_linear')
        })
    elif op_name == 'timestep_sample':
        details.update({
            'timestep_spacing': getattr(p, 'timestep_spacing', 'leading'),
            'timestep_respacing': getattr(p, 'timestep_respacing', '')
        })
    elif op_name.startswith('guidance_apply_'):
        step_num = op_name.split('_')[-1]
        details.update({
            'step_number': step_num,
            'guidance_scale': getattr(p, 'cfg_scale', 7.5),
            'guidance_rescale': getattr(p, 'guidance_rescale', 0.0)
        })
    
    # Control preprocessing operations
    elif op_name == 'canny_detect':
        details.update({
            'canny_low_threshold': getattr(p, 'canny_low_threshold', 100),
            'canny_high_threshold': getattr(p, 'canny_high_threshold', 200)
        })
    elif op_name == 'depth_estimate':
        details.update({
            'depth_estimator': getattr(p, 'depth_estimator', 'midas'),
            'depth_model_type': getattr(p, 'depth_model_type', 'dpt_large')
        })
    elif op_name == 'pose_detect':
        details.update({
            'pose_detector': getattr(p, 'pose_detector', 'openpose'),
            'pose_model': getattr(p, 'pose_model', 'body_25')
        })
    elif op_name == 'openpose_extract':
        details.update({
            'include_body': getattr(p, 'openpose_include_body', True),
            'include_face': getattr(p, 'openpose_include_face', True),
            'include_hand': getattr(p, 'openpose_include_hand', True)
        })
    elif op_name == 'segmentation':
        details.update({
            'segmentation_model': getattr(p, 'segmentation_model', 'sam'),
            'segmentation_classes': getattr(p, 'segmentation_classes', [])
        })
    elif op_name == 'controlnet_load':
        details.update({
            'controlnet_models': getattr(p, 'controlnet_models', []),
            'controlnet_weights': getattr(p, 'controlnet_weights', [])
        })
    
    # Text encoding operations
    elif op_name == 'text_tokenize':
        details.update({
            'prompt_length': len(getattr(p, 'prompt', '')),
            'negative_length': len(getattr(p, 'negative_prompt', '')),
            'max_tokens': 77,  # Default CLIP limit
        })
    elif op_name == 'text_encode':
        details.update({
            'encoder': 'CLIP' if hasattr(p, 'text_encoder') else 'Unknown',
            'clip_skip': getattr(p, 'clip_skip', 1),
            'prompt': getattr(p, 'prompt', '')[:50] + '...' if len(getattr(p, 'prompt', '')) > 50 else getattr(p, 'prompt', ''),
        })
    
    # VAE operations
    elif op_name == 'vae_encode':
        details.update({
            'vae_type': getattr(p, 'vae_type', 'Auto'),
            'input_size': f"{getattr(p, 'width', 0)}x{getattr(p, 'height', 0)}",
            'batch_size': getattr(p, 'batch_size', 1),
        })
    elif op_name == 'vae_decode':
        details.update({
            'vae_type': getattr(p, 'vae_type', 'Auto'),
            'output_size': f"{getattr(p, 'width', 0)}x{getattr(p, 'height', 0)}",
            'tiling': getattr(p, 'tiling', False),
        })
    
    # Generation operations
    elif op_name == 'noise_sample':
        details.update({
            'seed': getattr(p, 'seed', -1),
            'size': f"{getattr(p, 'width', 0)}x{getattr(p, 'height', 0)}",
            'scheduler': getattr(p, 'sampler_name', ''),
        })
    elif op_name.startswith('denoise_step_'):
        step_num = op_name.split('_')[-1]
        details.update({
            'step': f"{step_num}/{getattr(p, 'steps', 20)}",
            'cfg_scale': getattr(p, 'cfg_scale', 0),
            'scheduler': getattr(p, 'sampler_name', ''),
        })
    elif op_name == 'img2img_prep':
        details.update({
            'denoising': getattr(p, 'denoising_strength', 0),
            'init_images': len(getattr(p, 'init_images', [])),
            'size': f"{getattr(p, 'width', 0)}x{getattr(p, 'height', 0)}",
        })
    
    # High-res operations
    elif op_name == 'upscale':
        details.update({
            'upscaler': getattr(p, 'hr_upscaler', ''),
            'scale': getattr(p, 'hr_scale', 0),
            'target_size': f"{getattr(p, 'hr_resize_x', 0)}x{getattr(p, 'hr_resize_y', 0)}",
        })
    elif op_name == 'hires_denoise':
        details.update({
            'hr_steps': getattr(p, 'hr_second_pass_steps', 0),
            'hr_denoising': getattr(p, 'hr_denoising_strength', 0),
            'hr_scale': getattr(p, 'hr_scale', 0),
        })
    elif op_name == 'hires_decode':
        details.update({
            'output_size': f"{getattr(p, 'hr_resize_x', 0)}x{getattr(p, 'hr_resize_y', 0)}",
            'upscaler': getattr(p, 'hr_upscaler', ''),
        })
    
    # Control operations
    elif op_name == 'control_prep':
        details.update({
            'control_type': getattr(p, 'extra_generation_params', {}).get('Control type', ''),
            'control_model': getattr(p, 'extra_generation_params', {}).get('Control model', ''),
            'control_weight': getattr(p, 'extra_generation_params', {}).get('Control conditioning', '')
        })
    elif op_name == 'control':
        details.update({
            'control_type': getattr(p, 'extra_generation_params', {}).get('Control type', ''),
            'control_model': getattr(p, 'extra_generation_params', {}).get('Control model', ''),
            'control_weight': getattr(p, 'extra_generation_params', {}).get('Control conditioning', '')
        })
    
    # Post-processing operations
    elif op_name == 'face_detect':
        details.update({
            'detector': 'Auto',
            'confidence': 0.5,
        })
    elif op_name == 'face_restore':
        details.update({
            'detailer_steps': getattr(p, 'detailer_steps', 0),
            'detailer_strength': getattr(p, 'detailer_strength', 0),
            'model': 'GFPGAN/CodeFormer',
        })
    elif op_name == 'restore':
        details.update({
            'model': getattr(p, 'face_restoration_model', 'CodeFormer'),
            'strength': getattr(p, 'code_former_weight', 0.2),
        })
    elif op_name == 'detailer':
        details.update({
            'detailer_steps': getattr(p, 'detailer_steps', 0),
            'detailer_strength': getattr(p, 'detailer_strength', 0),
            'models': getattr(p, 'detailer_models', []),
        })
    elif op_name == 'color_correct' or op_name == 'color':
        details.update({
            'corrections': len(getattr(p, 'color_corrections', [])),
        })
    
    # Inpainting operations
    elif op_name == 'inpaint':
        details.update({
            'mask_blur': getattr(p, 'mask_blur', 0),
            'inpaint_full_res': getattr(p, 'inpaint_full_res', False),
            'inpainting_fill': getattr(p, 'inpainting_fill', 0),
        })
    
    # Instruct operations
    elif op_name == 'instruct':
        details.update({
            'instruction': getattr(p, 'prompt', ''),
            'image_cfg_scale': getattr(p, 'image_cfg_scale', 0),
        })
    
    # Video operations
    elif op_name == 'video':
        details.update({
            'frames': getattr(p, 'frames', 1),
            'fps': getattr(p, 'fps', 24),
            'video_type': getattr(p, 'video_type', 'None'),
        })
    
    # LCM operations
    elif op_name == 'lcm':
        details.update({
            'guidance_scale': getattr(p, 'cfg_scale', 0),
            'steps': getattr(p, 'steps', 0),
        })
    
    # Refiner operations
    elif op_name == 'refine':
        details.update({
            'refiner_steps': getattr(p, 'refiner_steps', 0),
            'refiner_start': getattr(p, 'refiner_start', 0),
            'refiner_prompt': getattr(p, 'refiner_prompt', ''),
        })
    
    # Legacy operations for backward compatibility
    elif op_name == 'txt2img':
        details.update({
            'prompt': getattr(p, 'prompt', ''),
            'negative_prompt': getattr(p, 'negative_prompt', ''),
            'steps': getattr(p, 'steps', 0),
            'cfg_scale': getattr(p, 'cfg_scale', 0),
            'sampler': getattr(p, 'sampler_name', ''),
            'size': f"{getattr(p, 'width', 0)}x{getattr(p, 'height', 0)}"
        })
    elif op_name == 'img2img':
        details.update({
            'prompt': getattr(p, 'prompt', ''),
            'negative_prompt': getattr(p, 'negative_prompt', ''),
            'steps': getattr(p, 'steps', 0),
            'cfg_scale': getattr(p, 'cfg_scale', 0),
            'sampler': getattr(p, 'sampler_name', ''),
            'size': f"{getattr(p, 'width', 0)}x{getattr(p, 'height', 0)}",
            'denoising': getattr(p, 'denoising_strength', 0),
        })
    elif op_name == 'hires':
        details.update({
            'hr_scale': getattr(p, 'hr_scale', 0),
            'hr_upscaler': getattr(p, 'hr_upscaler', ''),
            'hr_steps': getattr(p, 'hr_second_pass_steps', 0),
            'hr_denoising': getattr(p, 'hr_denoising_strength', 0)
        })
    
    return details


def setup_pipeline_operations(p):
    """Setup operations based on processing parameters with hierarchical structure"""
    if not pipeline_viz.enabled:
        return
        
    # Start new pipeline
    pipeline_id = f"pipeline_{int(time.time() * 1000)}"
    pipeline_viz.start_pipeline(pipeline_id)
    
    # Model loading operations (happening first)
    pipeline_viz.add_operation('model_load_text', 'Load Text Encoders')
    pipeline_viz.add_operation('model_load_unet', 'Load UNet Model', ['model_load_text'])
    pipeline_viz.add_operation('model_load_vae', 'Load VAE Model', ['model_load_unet'])
    pipeline_viz.add_operation('model_move_gpu', 'Move Models to GPU', ['model_load_vae'])
    
    # Memory management operations
    pipeline_viz.add_operation('memory_cleanup', 'Memory Cleanup')
    
    # Determine base operation type
    is_img2img = getattr(p, 'init_images', None) and len(p.init_images) > 0
    is_inpaint = getattr(p, 'mask', None) is not None or getattr(p, 'image_mask', None) is not None
    is_control = getattr(p, 'is_control', False)
    is_video = any(op in p.ops for op in ['video']) if hasattr(p, 'ops') else False
    
    # Base operation
    if is_video:
        base_op = 'video'
    elif is_inpaint:
        base_op = 'inpaint'
    elif is_img2img:
        base_op = 'img2img'
    else:
        base_op = 'txt2img'
    
    # Add base operation
    pipeline_viz.add_operation(base_op, base_op, ['model_move_gpu'])
    pipeline_viz.push_parent_operation(base_op)
    
    # Extension loading operations (LoRA, embeddings, etc.)
    extensions_loaded = []
    if hasattr(p, 'lora_modules') and p.lora_modules:
        pipeline_viz.add_operation('lora_load', 'Load LoRA Weights')
        pipeline_viz.add_operation('lora_apply', 'Apply LoRA to Model', ['lora_load'])
        extensions_loaded.append('lora_apply')
    
    if hasattr(p, 'embedding_names') and p.embedding_names:
        pipeline_viz.add_operation('embedding_load', 'Load Textual Inversions')
        extensions_loaded.append('embedding_load')
    
    if hasattr(p, 'ipadapter_model') and p.ipadapter_model:
        pipeline_viz.add_operation('ipadapter_apply', 'Apply IP-Adapter')
        extensions_loaded.append('ipadapter_apply')
    
    # Scheduler initialization
    pipeline_viz.add_operation('scheduler_init', 'Initialize Scheduler', extensions_loaded if extensions_loaded else [])
    
    # Memory optimization operations
    pipeline_viz.add_operation('attention_slicing', 'Attention Slicing Setup', ['scheduler_init'])
    pipeline_viz.add_operation('attention_chunking', 'Attention Chunking Setup', ['attention_slicing'])
    pipeline_viz.add_operation('gradient_checkpoint', 'Gradient Checkpointing', ['attention_chunking'])
    pipeline_viz.add_operation('mixed_precision', 'Mixed Precision Setup', ['gradient_checkpoint'])
    
    # Text processing sub-operations
    pipeline_viz.add_operation('text_tokenize', 'Text Tokenize', ['mixed_precision'])
    pipeline_viz.add_operation('text_encode', 'Text Encode', ['text_tokenize'])
    
    # Control preprocessing (if using ControlNet)
    if is_control:
        control_ops = []
        # Add control preprocessing operations
        if hasattr(p, 'control_types'):
            for control_type in p.control_types:
                if 'canny' in control_type.lower():
                    pipeline_viz.add_operation('canny_detect', 'Canny Edge Detection', ['text_encode'])
                    control_ops.append('canny_detect')
                elif 'depth' in control_type.lower():
                    pipeline_viz.add_operation('depth_estimate', 'Depth Estimation', ['text_encode'])
                    control_ops.append('depth_estimate')
                elif 'pose' in control_type.lower() or 'openpose' in control_type.lower():
                    pipeline_viz.add_operation('pose_detect', 'Pose Detection', ['text_encode'])
                    pipeline_viz.add_operation('openpose_extract', 'OpenPose Extraction', ['pose_detect'])
                    control_ops.append('openpose_extract')
                elif 'seg' in control_type.lower():
                    pipeline_viz.add_operation('segmentation', 'Segmentation Masks', ['text_encode'])
                    control_ops.append('segmentation')
        
        if control_ops:
            pipeline_viz.add_operation('controlnet_load', 'Load ControlNet Models', control_ops)
            text_deps = ['controlnet_load']
        else:
            text_deps = ['text_encode']
    else:
        text_deps = ['text_encode']
    
    # VAE operations
    if is_img2img or is_inpaint:
        pipeline_viz.add_operation('vae_encode', 'VAE Encode', text_deps)
        pipeline_viz.add_operation('img2img_prep', 'Image Preparation', ['vae_encode'])
        prep_op = 'img2img_prep'
    else:
        pipeline_viz.add_operation('noise_sample', 'Noise Sampling', text_deps)
        prep_op = 'noise_sample'
    
    # Noise scheduling operations
    pipeline_viz.add_operation('noise_schedule', 'Noise Schedule Setup', [prep_op])
    pipeline_viz.add_operation('timestep_sample', 'Timestep Sampling', ['noise_schedule'])
    
    # Individual denoising steps with guidance
    steps = getattr(p, 'steps', 20)
    prev_step = 'timestep_sample'
    for i in range(min(steps, 10)):  # Limit to 10 for UI clarity
        step_name = f'denoise_step_{i+1}'
        pipeline_viz.add_operation(step_name, f'Denoise Step {i+1}', [prev_step])
        
        # Add guidance application for each step
        guidance_name = f'guidance_apply_{i+1}'
        pipeline_viz.add_operation(guidance_name, f'CFG Guidance {i+1}', [step_name])
        prev_step = guidance_name
    
    # VAE decode
    pipeline_viz.add_operation('vae_decode', 'VAE Decode', [prev_step])
    last_op = 'vae_decode'
    
    # Memory offloading after main generation
    pipeline_viz.add_operation('model_offload', 'Model Offload', [last_op])
    pipeline_viz.add_operation('model_move_cpu', 'Move Models to CPU', ['model_offload'])
    
    # Pop base operation from stack
    pipeline_viz.pop_parent_operation()
    
    # High-res pass
    if getattr(p, 'enable_hr', False):
        pipeline_viz.add_operation('hires', 'High Resolution Pass', ['model_move_cpu'])
        pipeline_viz.push_parent_operation('hires')
        
        pipeline_viz.add_operation('upscale', 'Upscale')
        pipeline_viz.add_operation('hires_denoise', 'HR Denoise', ['upscale'])
        pipeline_viz.add_operation('hires_decode', 'HR Decode', ['hires_denoise'])
        
        pipeline_viz.pop_parent_operation()
        last_op = 'hires'
    else:
        last_op = 'model_move_cpu'
    
    # Control operations
    if is_control:
        pipeline_viz.add_operation('control', 'ControlNet', [last_op])
        pipeline_viz.push_parent_operation('control')
        
        pipeline_viz.add_operation('control_prep', 'Control Preparation')
        
        pipeline_viz.pop_parent_operation()
        last_op = 'control'
    
    # Refiner pass
    if getattr(p, 'refiner_steps', 0) > 0:
        pipeline_viz.add_operation('refine', 'Refiner Pass', [last_op])
        last_op = 'refine'
    
    # Post-processing operations
    post_ops = []
    
    # Face restoration
    if getattr(p, 'restore_faces', False):
        pipeline_viz.add_operation('restore', 'Face Restoration', [last_op])
        post_ops.append('restore')
    
    # Detailer
    if getattr(p, 'detailer_enabled', False):
        pipeline_viz.add_operation('detailer', 'Detailer', [last_op])
        pipeline_viz.push_parent_operation('detailer')
        
        pipeline_viz.add_operation('face_detect', 'Face Detection')
        pipeline_viz.add_operation('face_restore', 'Face Restore', ['face_detect'])
        
        pipeline_viz.pop_parent_operation()
        post_ops.append('detailer')
    
    # Color correction
    if hasattr(p, 'color_corrections') and p.color_corrections:
        deps = [post_ops[-1]] if post_ops else [last_op]
        pipeline_viz.add_operation('color_correct', 'Color Correction', deps)
        post_ops.append('color_correct')
    
    # Final memory cleanup
    final_deps = [post_ops[-1]] if post_ops else [last_op]
    pipeline_viz.add_operation('memory_cleanup_final', 'Final Memory Cleanup', final_deps)


def track_operation_start(op_name: str, p):
    """Track when an operation starts"""
    if not pipeline_viz.enabled:
        return
        
    details = get_operation_details(op_name, p)
    pipeline_viz.start_operation(op_name, details)


def track_operation_complete(op_name: str, p=None):
    """Track when an operation completes"""
    if not pipeline_viz.enabled:
        return
        
    details = {}
    if p:
        details = get_operation_details(op_name, p)
    pipeline_viz.complete_operation(op_name, details)


def track_operation_fail(op_name: str, error: str):
    """Track when an operation fails"""
    if not pipeline_viz.enabled:
        return
        
    pipeline_viz.fail_operation(op_name, error)


def get_current_pipeline_state():
    """Get current pipeline state for API/UI"""
    return pipeline_viz.get_pipeline_state()


def track_sub_operation_start(op_name: str, parent_op: str, p):
    """Track when a sub-operation starts"""
    if not pipeline_viz.enabled:
        return
        
    pipeline_viz.add_operation(op_name, op_name, parent=parent_op)
    details = get_operation_details(op_name, p)
    pipeline_viz.start_operation(op_name, details)


def track_sub_operation_complete(op_name: str, p=None):
    """Track when a sub-operation completes"""
    if not pipeline_viz.enabled:
        return
        
    details = {}
    if p:
        details = get_operation_details(op_name, p)
    pipeline_viz.complete_operation(op_name, details) 