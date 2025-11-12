import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal, List, Tuple

from pyhealth.models import BaseModel


class LayerWiseRelevancePropagation:
    """Layer-wise Relevance Propagation attribution method for PyHealth models.

    This class implements the LRP method for computing feature attributions
    in neural networks. The method decomposes the network's prediction into
    relevance scores for each input feature through backward propagation of
    relevance from output to input layers.

    The method is based on the paper:
        Layer-wise Relevance Propagation for Neural Networks with
        Local Renormalization Layers
        Alexander Binder, Gregoire Montavon, Sebastian Bach,
        Klaus-Robert Muller, Wojciech Samek
        arXiv:1604.00825, 2016
        https://arxiv.org/abs/1604.00825

    LRP satisfies the conservation property: relevance is conserved at
    each layer, meaning the sum of relevances at the input layer equals
    the model's output for the target class.

    Key differences from Integrated Gradients:
        - LRP: Single backward pass, no baseline needed, sums to f(x)
        - IG: Multiple forward passes, requires baseline, sums to f(x)-f(baseline)

    Args:
        model (BaseModel): A trained PyHealth model to interpret. Must have
            been trained and should be in evaluation mode.
        rule (str): LRP propagation rule to use:
            - "epsilon": ε-rule for numerical stability (default)
            - "alphabeta": αβ-rule for sharper visualizations
        epsilon (float): Stabilizer for ε-rule. Default 0.01.
            Prevents division by zero in relevance redistribution.
        alpha (float): α parameter for αβ-rule. Default 1.0.
            Controls positive contribution weighting.
        beta (float): β parameter for αβ-rule. Default 0.0.
            Controls negative contribution weighting.
        use_embeddings (bool): If True, compute relevance from embedding
            layer for models with discrete inputs. Default True.
            Required for models with discrete medical codes.

    Note:
        This implementation currently supports:
        - Linear layers (fully connected)
        - ReLU activations
        - Embedding layers
        - Basic sequential models (MLP, simple RNN)

        Future versions will add support for:
        - Attention mechanisms
        - Complex temporal models (StageNet)
        - Normalization layers
        - Pooling operations

    Examples:
        >>> from pyhealth.interpret.methods import LayerWiseRelevancePropagation
        >>> from pyhealth.models import MLP
        >>> from pyhealth.datasets import get_dataloader
        >>>
        >>> # Initialize LRP with trained model
        >>> lrp = LayerWiseRelevancePropagation(
        ...     model=trained_model,
        ...     rule="epsilon",
        ...     epsilon=0.01
        ... )
        >>>
        >>> # Get test data
        >>> test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
        >>> test_batch = next(iter(test_loader))
        >>>
        >>> # Compute attributions
        >>> attributions = lrp.attribute(**test_batch)
        >>>
        >>> # Print results
        >>> for feature_key, relevance in attributions.items():
        ...     print(f"{feature_key}: shape={relevance.shape}")
        ...     print(f"  Sum of relevances: {relevance.sum().item():.4f}")
        ...     print(f"  Top 5 indices: {relevance.flatten().topk(5).indices}")
        >>>
        >>> # Use αβ-rule for sharper heatmaps
        >>> lrp_sharp = LayerWiseRelevancePropagation(
        ...     model=trained_model,
        ...     rule="alphabeta",
        ...     alpha=1.0,
        ...     beta=0.0
        ... )
        >>> sharp_attrs = lrp_sharp.attribute(**test_batch)
    """

    def __init__(
        self,
        model: BaseModel,
        rule: Literal["epsilon", "alphabeta"] = "epsilon",
        epsilon: float = 0.01,
        alpha: float = 1.0,
        beta: float = 0.0,
        use_embeddings: bool = True,
    ):
        """Initialize LRP interpreter.

        Args:
            model: A trained PyHealth model to interpret.
            rule: Propagation rule ("epsilon" or "alphabeta").
            epsilon: Stabilizer for epsilon-rule.
            alpha: Alpha parameter for alphabeta-rule.
            beta: Beta parameter for alphabeta-rule.
            use_embeddings: Whether to start from embedding layer.

        Raises:
            AssertionError: If use_embeddings=True but model does not
                implement forward_from_embedding() method.
        """
        self.model = model
        self.model.eval()  # Ensure model is in evaluation mode
        self.rule = rule
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.use_embeddings = use_embeddings

        # Storage for activations and hooks
        self.hooks = []
        self.activations = {}
        
        # Track concatenation structure for branching models
        self.branch_info = {}  # Maps layer names to their feature branches
        self.concat_point = None  # Name of the layer where concatenation occurs

        # Validate model compatibility
        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() method to support embedding-level "
                "LRP. Set use_embeddings=False to use input-level LRP "
                "(only for continuous features)."
            )

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP attributions for input features.

        This method computes relevance scores by:
        1. Performing a forward pass to get the prediction
        2. Initializing output layer relevance
        3. Propagating relevance backward through layers
        4. Mapping relevance to input features

        Args:
            target_class_idx: Target class index for attribution
                computation. If None, uses the predicted class (argmax of
                model output).
            **data: Input data dictionary from a dataloader batch
                containing:
                - Feature keys (e.g., 'conditions', 'procedures'):
                  Input tensors for each modality
                - 'label' (optional): Ground truth label tensor
                - Other metadata keys are ignored

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping each feature key
                to its relevance tensor. Each tensor has the same shape
                as the input tensor, with values indicating the
                contribution of each input element to the model's
                prediction.

                Positive values indicate features that increase the
                prediction score, while negative values indicate features
                that decrease it.

                Important: Unlike Integrated Gradients, LRP relevances
                sum to approximately f(x) (the model's output), not to
                f(x) - f(baseline).

        Note:
            - Relevance conservation: Sum of input relevances should
              approximately equal the model's output for the target class.
            - For better interpretability, use batch_size=1 or analyze
              samples individually.
            - The quality of attributions depends on the chosen rule and
              parameters (epsilon, alpha, beta).

        Examples:
            >>> # Basic usage with default settings
            >>> attributions = lrp.attribute(**test_batch)
            >>> print(f"Total relevance: {sum(r.sum() for r in attributions.values())}")
            >>>
            >>> # Specify target class explicitly
            >>> attributions = lrp.attribute(**test_batch, target_class_idx=1)
            >>>
            >>> # Analyze which features are most important
            >>> condition_relevance = attributions['conditions'][0]
            >>> top_k = torch.topk(condition_relevance.flatten(), k=5)
            >>> print(f'Most relevant features: {top_k.indices}')
            >>> print(f'Relevance values: {top_k.values}')
        """
        # Extract feature keys and prepare inputs
        feature_keys = self.model.feature_keys
        inputs = {}
        time_info = {}  # Store time information for StageNet-like models
        label_data = {}  # Store label information

        # Process input features
        for key in feature_keys:
            if key in data:
                x = data[key]
                # Handle tuple inputs (e.g., StageNet with (time, values))
                if isinstance(x, tuple):
                    time_info[key] = x[0]  # Store time component
                    x = x[1]  # Use values component for attribution

                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)

                x = x.to(next(self.model.parameters()).device)
                inputs[key] = x

        # Store label data for passing to model
        for key in self.model.label_keys:
            if key in data:
                label_val = data[key]
                if not isinstance(label_val, torch.Tensor):
                    label_val = torch.tensor(label_val)
                label_val = label_val.to(next(self.model.parameters()).device)
                label_data[key] = label_val

        # Compute LRP attributions
        if self.use_embeddings:
            attributions = self._compute_from_embeddings(
                inputs=inputs,
                target_class_idx=target_class_idx,
                time_info=time_info,
                label_data=label_data,
            )
        else:
            # Direct input-level LRP (for continuous features only)
            raise NotImplementedError(
                "Input-level LRP not yet implemented. "
                "Use use_embeddings=True for models with discrete features."
            )

        return attributions

    def _compute_from_embeddings(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class_idx: Optional[int] = None,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        label_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP starting from embedding layer.

        This method:
        1. Embeds discrete inputs into continuous space
        2. Performs forward pass while capturing activations
        3. Initializes relevance at output layer
        4. Propagates relevance backward to embeddings
        5. Maps relevance back to input tokens

        Args:
            inputs: Dictionary of input tensors for each feature.
            target_class_idx: Target class for attribution.
            time_info: Optional time information for temporal models.
            label_data: Optional label data to pass to model.

        Returns:
            Dictionary of relevance scores per feature.
        """
        # Step 1: Embed inputs using model's embedding layer
        input_embeddings = {}
        input_shapes = {}  # Store original shapes for later mapping

        for key in inputs:
            input_shapes[key] = inputs[key].shape
            # Get embeddings from model's embedding layer
            embedded = self.model.embedding_model({key: inputs[key]})
            x = embedded[key]

            # Handle nested sequences (4D tensors) by pooling
            if x.dim() == 4:  # [batch, seq_len, tokens, embedding_dim]
                # Sum pool over inner dimension
                x = x.sum(dim=2)  # [batch, seq_len, embedding_dim]

            input_embeddings[key] = x

        # Step 2: Register hooks to capture activations during forward pass
        self._register_hooks()

        try:
            # Step 3: Forward pass through model
            forward_kwargs = {**label_data} if label_data else {}

            with torch.no_grad():
                output = self.model.forward_from_embedding(
                    feature_embeddings=input_embeddings,
                    time_info=time_info,
                    **forward_kwargs,
                )
            logits = output["logit"]

            # Step 4: Determine target class
            if target_class_idx is None:
                target_class_idx = torch.argmax(logits, dim=-1)
            elif not isinstance(target_class_idx, torch.Tensor):
                target_class_idx = torch.tensor(
                    target_class_idx, device=logits.device
                )

            # Step 5: Initialize output relevance
            # For classification: start with the target class output
            if logits.dim() == 2 and logits.size(-1) > 1:
                # Multi-class: one-hot encoding
                batch_size = logits.size(0)
                output_relevance = torch.zeros_like(logits)
                output_relevance[range(batch_size), target_class_idx] = logits[
                    range(batch_size), target_class_idx
                ]
            else:
                # Binary classification
                output_relevance = logits

            # Step 6: Propagate relevance backward through network
            relevance_at_embeddings = self._propagate_relevance_backward(
                output_relevance, input_embeddings
            )

            # Step 7: Map relevance back to input space
            input_relevances = {}
            for key in input_embeddings:
                rel = relevance_at_embeddings.get(key)
                if rel is not None:
                    # Sum over embedding dimension to get per-token relevance
                    if rel.dim() == 3:  # [batch, seq_len, embedding_dim]
                        input_relevances[key] = rel.sum(dim=-1)  # [batch, seq_len]
                    elif rel.dim() == 2:  # [batch, embedding_dim]
                        input_relevances[key] = rel.sum(dim=-1)  # [batch]
                    else:
                        input_relevances[key] = rel

                    # Expand to match original input shape if needed
                    orig_shape = input_shapes[key]
                    if input_relevances[key].shape != orig_shape:
                        # Handle case where input was 3D but we have 2D relevance
                        if len(orig_shape) == 3 and input_relevances[key].dim() == 2:
                            # Broadcast to match
                            input_relevances[key] = input_relevances[key].unsqueeze(
                                -1
                            ).expand(orig_shape)

        finally:
            # Step 8: Clean up hooks
            self._remove_hooks()

        return input_relevances

    def _register_hooks(self):
        """Register forward hooks to capture activations during forward pass.

        Hooks are attached to all relevant layer types to capture both
        inputs and outputs for later relevance propagation.
        
        Also detects branching structure (e.g., ModuleDict with parallel branches).
        """

        def save_activation(name):
            def hook(module, input, output):
                # Store both input and output activations
                # Handle tuple inputs (e.g., from LSTM)
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input

                self.activations[name] = {
                    "input": input_tensor,
                    "output": output,
                    "module": module,
                }

            return hook

        # Detect branching structure for models with ModuleDict (like MLP)
        self._detect_branches()
        
        # Register hooks on layers we can propagate through
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.LSTM, nn.GRU)):
                handle = module.register_forward_hook(save_activation(name))
                self.hooks.append(handle)

    def _detect_branches(self):
        """Detect parallel branches in model architecture (e.g., ModuleDict in MLP).
        
        For PyHealth MLP models, features are processed in parallel branches
        through separate MLPs, then concatenated before the final FC layer.
        This method identifies these branches and the concatenation point.
        """
        # Check if model has feature_keys attribute (indicating MLP-like structure)
        if hasattr(self.model, 'feature_keys') and hasattr(self.model, 'mlp'):
            # Model has parallel branches per feature
            for feature_key in self.model.feature_keys:
                branch_layers = []
                # Find all layers in this branch
                for name, module in self.model.named_modules():
                    if f'mlp.{feature_key}' in name:
                        branch_layers.append(name)
                
                if branch_layers:
                    self.branch_info[feature_key] = {
                        'layers': branch_layers,
                        'hidden_dim': getattr(self.model, 'hidden_dim', None)
                    }
            
            # The FC layer is where concatenation outputs are processed
            if hasattr(self.model, 'fc'):
                self.concat_point = 'fc'
    
    def _remove_hooks(self):
        """Remove all registered hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.branch_info = {}
        self.concat_point = None

    def _propagate_relevance_backward(
        self,
        output_relevance: torch.Tensor,
        input_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Propagate relevance from output layer back to input embeddings.

        This is the core LRP algorithm. It iterates through layers in
        reverse order, applying the appropriate LRP rule to redistribute
        relevance from each layer to the previous layer.
        
        For branching architectures (like PyHealth MLP), this handles:
        1. Propagating from FC layer to concatenation point
        2. Splitting relevance for parallel branches
        3. Propagating through each branch independently
        4. Collecting relevance at embeddings

        Args:
            output_relevance: Relevance at the output layer.
            input_embeddings: Dictionary of input embeddings for each feature.

        Returns:
            Dictionary of relevance scores at the embedding layer.
        """
        # Start with output relevance
        current_relevance = output_relevance

        # Get layer names in reverse order (output to input)
        layer_names = list(reversed(list(self.activations.keys())))
        
        # Track relevance for each branch if we have branching architecture
        branch_relevances = {}
        in_branches = False

        # Propagate through each layer
        for idx, layer_name in enumerate(layer_names):
            activation_info = self.activations[layer_name]
            module = activation_info["module"]
            
            # Check if this is the concatenation point (FC layer in MLP)
            if self.concat_point and layer_name == self.concat_point:
                # Propagate through FC layer
                if isinstance(module, nn.Linear):
                    current_relevance = self._lrp_linear(
                        module, activation_info, current_relevance
                    )
                
                # After FC, split relevance for parallel branches
                if self.branch_info:
                    branch_relevances = self._split_relevance_for_branches(
                        current_relevance, input_embeddings
                    )
                    in_branches = True
                continue
            
            # If we're in parallel branches, process each branch separately
            if in_branches and self.branch_info:
                # Check which branch this layer belongs to
                for feature_key, branch_data in self.branch_info.items():
                    if layer_name in branch_data['layers']:
                        # Propagate through this branch layer
                        if isinstance(module, nn.Linear):
                            branch_relevances[feature_key] = self._lrp_linear(
                                module, activation_info, branch_relevances[feature_key]
                            )
                        elif isinstance(module, nn.ReLU):
                            branch_relevances[feature_key] = self._lrp_relu(
                                activation_info, branch_relevances[feature_key]
                            )
                        break
            else:
                # Standard sequential propagation
                if isinstance(module, nn.Linear):
                    current_relevance = self._lrp_linear(
                        module, activation_info, current_relevance
                    )
                elif isinstance(module, nn.ReLU):
                    current_relevance = self._lrp_relu(activation_info, current_relevance)
                elif isinstance(module, (nn.LSTM, nn.GRU)):
                    current_relevance = self._lrp_rnn(
                        module, activation_info, current_relevance
                    )

        # If we have branch relevances, use those; otherwise use current_relevance
        if branch_relevances:
            return branch_relevances
        else:
            # No branching - split the relevance to features
            return self._split_relevance_to_features(current_relevance, input_embeddings)

    def _lrp_linear(
        self,
        module: nn.Linear,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply LRP to a linear (fully connected) layer.

        Uses either epsilon-rule or alphabeta-rule depending on
        initialization.

        Args:
            module: The linear layer.
            activation_info: Dictionary containing input/output activations.
            relevance_output: Relevance from the next layer.

        Returns:
            Relevance for the previous layer.
        """
        if self.rule == "epsilon":
            return self._lrp_linear_epsilon(module, activation_info, relevance_output)
        elif self.rule == "alphabeta":
            return self._lrp_linear_alphabeta(
                module, activation_info, relevance_output
            )
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

    def _lrp_linear_epsilon(
        self,
        module: nn.Linear,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP epsilon-rule for linear layers.

        Formula: R_i = Σ_j (z_ij / (z_j + ε·sign(z_j))) · R_j

        where z_ij = w_ij * x_i and z_j = Σ_k w_kj * x_k + b_j

        Args:
            module: The linear layer.
            activation_info: Stored activations.
            relevance_output: Relevance from next layer.

        Returns:
            Relevance for previous layer.
        """
        # Get input activations
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]

        # Ensure x is 2D [batch, features]
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)

        # Get weights and bias
        W = module.weight  # [out_features, in_features]
        b = module.bias if module.bias is not None else 0

        # Compute z_j = W @ x^T + b
        z = F.linear(x, W, b)  # [batch, out_features]

        # Add stabilizer to prevent division by zero
        z_stabilized = z + self.epsilon * torch.sign(z)
        z_stabilized = torch.where(
            torch.abs(z_stabilized) < 1e-9,
            torch.ones_like(z_stabilized) * 1e-9,
            z_stabilized,
        )

        # Compute relevance for each input
        # R_i = Σ_j (w_ij * x_i / z_j) * R_j
        # Reshape for broadcasting
        x_expanded = x.unsqueeze(2)  # [batch, in_features, 1]
        W_expanded = W.t().unsqueeze(0)  # [1, in_features, out_features]
        z_expanded = z_stabilized.unsqueeze(1)  # [batch, 1, out_features]
        R_expanded = relevance_output.unsqueeze(1)  # [batch, 1, out_features]

        # Compute z_ij = w_ij * x_i
        z_ij = x_expanded * W_expanded  # [batch, in_features, out_features]

        # Compute relevance redistribution
        relevance_input = ((z_ij / z_expanded) * R_expanded).sum(
            dim=2
        )  # [batch, in_features]

        return relevance_input

    def _lrp_linear_alphabeta(
        self,
        module: nn.Linear,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP alphabeta-rule for linear layers.

        Formula: R_i = Σ_j [(α·z_ij^+ / z_j^+) - (β·z_ij^- / z_j^-)] · R_j

        This rule separates positive and negative contributions.
        Common setting: α=1, β=0 (ignore negative contributions).

        Args:
            module: The linear layer.
            activation_info: Stored activations.
            relevance_output: Relevance from next layer.

        Returns:
            Relevance for previous layer.
        """
        # Get input activations
        x = activation_info["input"]
        if isinstance(x, tuple):
            x = x[0]

        # Ensure x is 2D
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)

        # Get weights and bias
        W = module.weight
        b = module.bias if module.bias is not None else 0

        # Separate positive and negative weights
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)

        # Separate positive and negative bias
        if isinstance(b, torch.Tensor):
            b_pos = torch.clamp(b, min=0)
            b_neg = torch.clamp(b, max=0)
        else:
            b_pos = 0
            b_neg = 0

        # Compute positive and negative forward passes
        z_pos = F.linear(x, W_pos, b_pos)  # [batch, out_features]
        z_neg = F.linear(x, W_neg, b_neg)  # [batch, out_features]

        # Add small epsilon to avoid division by zero
        eps = 1e-9
        z_pos = torch.where(
            torch.abs(z_pos) < eps, torch.ones_like(z_pos) * eps, z_pos
        )
        z_neg = torch.where(
            torch.abs(z_neg) < eps, torch.ones_like(z_neg) * eps, z_neg
        )

        # Expand for broadcasting
        x_expanded = x.unsqueeze(2)  # [batch, in_features, 1]
        W_pos_expanded = W_pos.t().unsqueeze(0)  # [1, in_features, out_features]
        W_neg_expanded = W_neg.t().unsqueeze(0)  # [1, in_features, out_features]
        z_pos_expanded = z_pos.unsqueeze(1)  # [batch, 1, out_features]
        z_neg_expanded = z_neg.unsqueeze(1)  # [batch, 1, out_features]
        R_expanded = relevance_output.unsqueeze(1)  # [batch, 1, out_features]

        # Compute positive and negative contributions
        z_ij_pos = x_expanded * W_pos_expanded
        z_ij_neg = x_expanded * W_neg_expanded

        # Compute relevance
        relevance_pos = ((z_ij_pos / z_pos_expanded) * R_expanded).sum(dim=2)
        relevance_neg = ((z_ij_neg / z_neg_expanded) * R_expanded).sum(dim=2)

        relevance_input = self.alpha * relevance_pos - self.beta * relevance_neg

        return relevance_input

    def _lrp_relu(
        self, activation_info: dict, relevance_output: torch.Tensor
    ) -> torch.Tensor:
        """LRP for ReLU activation.

        ReLU is element-wise, so relevance passes through unchanged.
        Only positive activations contributed to the output.

        Args:
            activation_info: Stored activations.
            relevance_output: Relevance from next layer.

        Returns:
            Relevance for previous layer (unchanged).
        """
        # ReLU doesn't change relevance distribution
        # Relevance flows through unchanged
        return relevance_output

    def _lrp_rnn(
        self,
        module: nn.Module,
        activation_info: dict,
        relevance_output: torch.Tensor,
    ) -> torch.Tensor:
        """LRP for RNN/LSTM/GRU layers.

        This is a simplified approach that treats the RNN as a black box.
        For more sophisticated temporal LRP, see LRP-LSTM papers.

        Args:
            module: The RNN layer.
            activation_info: Stored activations.
            relevance_output: Relevance from next layer.

        Returns:
            Relevance for previous layer.
        """
        # Simplified: distribute relevance uniformly over time steps
        # More sophisticated approaches would consider hidden states

        input_tensor = activation_info["input"]
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]

        # For now, assume relevance_output is [batch, hidden_size]
        # and input is [batch, seq_len, input_size]

        if input_tensor.dim() == 3:
            batch_size, seq_len, input_size = input_tensor.shape
            # Distribute relevance equally across time steps
            # This is a simplification - real LRP for RNN is more complex
            relevance_per_timestep = relevance_output.unsqueeze(1).expand(
                batch_size, seq_len, -1
            )
            return relevance_per_timestep
        else:
            return relevance_output

    def _split_relevance_for_branches(
        self,
        relevance: torch.Tensor,
        input_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Split relevance from concatenated layer to individual feature branches.
        
        When multiple features are processed in parallel and then concatenated
        (as in PyHealth MLP), we need to split the relevance proportionally
        back to each branch.
        
        Args:
            relevance: Relevance tensor from concatenated layer [batch, total_hidden_dim]
            input_embeddings: Dictionary of embeddings for each feature
            
        Returns:
            Dictionary mapping feature keys to their relevance tensors
        """
        branch_relevances = {}
        
        if not self.branch_info:
            # No branching detected, return as-is
            return self._split_relevance_to_features(relevance, input_embeddings)
        
        # Split relevance based on hidden dimensions of each branch
        # In MLP, each feature gets hidden_dim outputs that are concatenated
        offset = 0
        for feature_key in self.model.feature_keys:
            if feature_key in self.branch_info:
                hidden_dim = self.branch_info[feature_key]['hidden_dim']
                if hidden_dim is None:
                    # Fallback: equal split
                    hidden_dim = relevance.shape[-1] // len(self.model.feature_keys)
                
                # Extract this branch's portion of relevance
                branch_relevances[feature_key] = relevance[..., offset:offset + hidden_dim]
                offset += hidden_dim
        
        return branch_relevances
    
    def _split_relevance_to_features(
        self,
        relevance: torch.Tensor,
        input_embeddings: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Split combined relevance back to individual features.

        In PyHealth models, embeddings from different features are
        concatenated before final classification. This method splits
        the relevance back to each feature.

        Args:
            combined_relevance: Relevance at concatenated embedding layer.
            input_embeddings: Original input embeddings for each feature.

        Returns:
            Dictionary mapping feature keys to their relevance tensors.
        """
        relevance_by_feature = {}

        # Calculate the size of each feature's embedding
        feature_sizes = {}
        for key, emb in input_embeddings.items():
            if emb.dim() == 3:  # [batch, seq_len, embedding_dim]
                feature_sizes[key] = emb.size(1) * emb.size(2)
            elif emb.dim() == 2:  # [batch, embedding_dim]
                feature_sizes[key] = emb.size(1)
            else:
                feature_sizes[key] = emb.numel() // emb.size(0)

        # Split relevance according to feature sizes
        # This assumes features are concatenated in order of feature_keys
        if relevance.dim() == 2:  # [batch, total_features]
            current_idx = 0
            for key in self.model.feature_keys:
                if key in input_embeddings:
                    size = feature_sizes[key]
                    rel_chunk = relevance[:, current_idx : current_idx + size]

                    # Reshape to match original embedding shape
                    emb_shape = input_embeddings[key].shape
                    if len(emb_shape) == 3:
                        rel_chunk = rel_chunk.view(
                            emb_shape[0], emb_shape[1], emb_shape[2]
                        )
                    elif len(emb_shape) == 2:
                        rel_chunk = rel_chunk.view(emb_shape[0], emb_shape[1])

                    relevance_by_feature[key] = rel_chunk
                    current_idx += size
        else:
            # If relevance doesn't match expected shape, return as-is for each feature
            # This is a fallback for complex architectures
            for key in input_embeddings:
                relevance_by_feature[key] = relevance

        return relevance_by_feature
