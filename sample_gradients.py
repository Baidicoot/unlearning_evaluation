import torch
import os
from typing import Union, Optional, Dict, Tuple
from contextlib import contextmanager
import numpy as np
import random
import tqdm

from enum import Enum, auto

class ProjectionMethod(Enum):
    DENSE = auto()  # Dense random projection matrix
    SPARSE = auto()  # Sparse random projection matrix 
    SUBSAMPLE = auto()  # Subsample gradients directly

class GradientProjector:
    """Handles seeded random projection and efficient saving of gradient updates."""
    
    def __init__(
        self,
        projection_dim: int = 1000,
        projection_method: ProjectionMethod = ProjectionMethod.DENSE,
        seed: int = 42,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            projection_dim: Dimension to project gradients into
            projection_method: Method to project gradients
            seed: Random seed for reproducible projections
            device: Device to store projection matrix on
            torch_dtype: Data type to store projection matrix on
        """
        self.projection_dim = projection_dim
        self.projection_method = projection_method
        self.seed = seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection_matrices = {}  # Store matrices per input dimension
        self.torch_dtype = torch_dtype
        
    @contextmanager
    def _temp_seed(self, input_dim: int) -> None:
        """Temporarily set random seed for matrix generation."""
        # Store current random states
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        try:
            # Set seeds for reproducible matrix generation
            # Use input_dim in seed to ensure different matrices for different dims
            full_seed = hash((self.seed, input_dim)) % (2**32)
            random.seed(full_seed)
            np.random.seed(full_seed)
            torch.manual_seed(full_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(full_seed)
            yield
        finally:
            # Restore previous random states
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
            
    def _get_projection_matrix(self, input_dim: int) -> torch.Tensor:
        """Get or create seeded random projection matrix for given input dimension."""
        assert self.projection_method in [ProjectionMethod.DENSE, ProjectionMethod.SPARSE]

        if input_dim not in self.projection_matrices:
            with self._temp_seed(input_dim):
                if self.projection_method == ProjectionMethod.SPARSE:
                    # Create sparse random projection matrix (memory efficient)
                    density = 1 / np.sqrt(input_dim)  # Theoretical optimal density
                    mask = torch.bernoulli(torch.full((self.projection_dim, input_dim), density))
                    values = torch.randn(int(mask.sum().item())) * np.sqrt(1 / density)
                    matrix = torch.sparse_coo_tensor(
                        mask.nonzero().t(),
                        values,
                        mask.size(),
                        device=self.device,
                        dtype=self.torch_dtype
                    )
                else:
                    # Dense random projection matrix
                    matrix = torch.randn(
                        self.projection_dim,
                        input_dim,
                        device=self.device,
                        dtype=self.torch_dtype
                    ) / np.sqrt(self.projection_dim)
                
                self.projection_matrices[input_dim] = matrix
                
        return self.projection_matrices[input_dim]

    def _project_single_gradient(
        self,
        grad: torch.Tensor
    ) -> torch.Tensor:
        """Project a single gradient tensor."""
        grad_flat = grad.detach().view(-1)
        size = grad_flat.size(0)

        if self.projection_method == ProjectionMethod.SUBSAMPLE:
            with self._temp_seed(size):
                return grad_flat[torch.randperm(size)[:self.projection_dim]]
        else:
            matrix = self._get_projection_matrix(size)
            
            if self.projection_method == ProjectionMethod.SPARSE:
                return torch.sparse.mm(matrix, grad_flat.unsqueeze(1)).squeeze()
            else:
                return matrix @ grad_flat

    def project_gradients(
        self,
        model: torch.nn.Module,
        concat_results: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Project gradients from all parameters into lower dimensional space.
        Uses component-wise projection to avoid memory issues with large models.
        
        Args:
            model: PyTorch model with gradients
            concat_results: Whether to concatenate projected results
            
        Returns:
            Projected gradients either as single tensor or dict of tensors
        """
        # Collect and project gradients
        param_names = {n: p for n, p in model.named_parameters() if p.grad is not None}

        projected_grads = {}
        for i, (name, param) in tqdm.tqdm(enumerate(sorted(param_names.items())), total=len(param_names)):  # Sort for deterministic ordering
            projected_grads[name] = self._project_single_gradient(param.grad)
        
        if concat_results:
            # Concatenate the already-projected gradients
            return torch.cat([proj for proj in projected_grads.values()])
        else:
            return projected_grads

    @staticmethod
    def save_projected_gradients(
        projected_grads: Union[torch.Tensor, Dict[str, torch.Tensor]],
        filepath: str,
        compressed: bool = True
    ) -> None:
        """
        Efficiently save projected gradients to disk.
        
        Args:
            projected_grads: Projected gradients from project_gradients()
            filepath: Path to save file
            compressed: Whether to use compression
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if compressed:
            torch.save(
                projected_grads,
                filepath,
                _use_new_zipfile_serialization=True,
                pickle_protocol=4
            )
        else:
            torch.save(projected_grads, filepath)

@contextmanager
def track_gradient_projection(
    model: torch.nn.Module,
    save_path: str,
    projection_dim: int = 1000,
    projection_method: ProjectionMethod = ProjectionMethod.DENSE,
    seed: int = 42,
    torch_dtype: Optional[torch.dtype] = None,
    compressed: bool = True
):
    """Context manager for easy gradient projection and saving.
    
    Example:
        with track_gradient_projection(model, "gradients/step_1.pt", seed=42):
            loss = criterion(model(x), y)
            loss.backward()
    """
    projector = GradientProjector(projection_dim, projection_method, seed, torch_dtype=torch_dtype)
    yield
    
    projected_grads = projector.project_gradients(model)
    projector.save_projected_gradients(projected_grads, save_path, compressed)