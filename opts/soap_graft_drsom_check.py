import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Iterable
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import math

from itertools import chain

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py

ParamsT = Iterable[nn.Parameter]

__all__ = ["SOAP"]

class SOAP_drsom_check(optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        max_precond_dim (`int`, *optional*, defaults to 10000):
            Maximum dimension of the preconditioner.
            Set to 10000, so that we exclude most common vocab sizes while including layers.
        merge_dims (`bool`, *optional*, defaults to `False`):
            Whether or not to merge dimensions of the preconditioner.
        precondition_1d (`bool`, *optional*, defaults to `False`):
            Whether or not to precondition 1D gradients.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
            Helps at large precondition_frequency (~100 in our experiments), 
            but hurts performance at small precondition_frequency (~10 in our experiments).
        data_format (`str`, *optional*, defaults to `channels_first`):
            Data format of the input for convolutional layers.
            Should be "channels_last" for data_format of NHWC and "channels_first" for NCHW.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias correction in Adam.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float= -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int=1000,
        max_precond_dim: int=10000, # 
        merge_dims: bool = False, # Merge dimensions till the product of the dimensions is less than or equal to max_precond_dim.
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format
        
    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        
        if curr_shape > 1 or len(new_shape)==0:
            new_shape.append(curr_shape)
        
        new_grad = grad.reshape(new_shape)
        return new_grad               


    @torch.no_grad()
    def step(self, closure = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            closure = torch.enable_grad()(closure)
            loss = closure()
        
        for group in self.param_groups:
            # print("Processing group with params:", [p.shape for p in group["params"]])
            for p in group["params"]:
                # print(f"Processing parameter {p.shape} with grad {p.grad.shape} and state {self.state[p].keys()}")
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0 

                prev_d = state.get("prev_d", None) # The previous momentum direction with the same shape as grad.
                tolerance_grad = group["eps"]
                
                if  ('Q' not in state):
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group['precondition_frequency'],
                        precondition_1d=group['precondition_1d'],
                        shampoo_beta=(group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]),
                        max_precond_dim=group['max_precond_dim'],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(grad, state,
                                               max_precond_dim=group['max_precond_dim'],
                                               merge_dims=group["merge_dims"],
                                               precondition_1d=group["precondition_1d"])
                    d = -grad.detach().clone() * 1e-4
                    state["prev_d"] = d.detach().clone() # Save the current momentum direction for the next step.
                    # print("Initialized preconditioner.")
                    continue # first step is skipped so that we never use the current gradients in the projection.
                
                # Projecting gradients to the eigenbases of Shampoo's preconditioner 
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = self.project(grad, state, merge_dims=group["merge_dims"], 
                                              max_precond_dim=group['max_precond_dim'])
                # print(f"grad_projected.shape = {grad_projected.shape}, grad.shape = {grad.shape}")
                if prev_d is not None:
                    mom_projected = self.project(prev_d , state , merge_dims=group["merge_dims"], max_precond_dim=group['max_precond_dim'])
                else:
                    mom_projected = None # If no previous momentum direction, use the negative gradient as the initial momentum direction.

                if prev_d is not None:
                    with torch.enable_grad():
                        # print("prev_d.shape = ", prev_d.shape)
                        grad_requires_grad = torch.autograd.grad(loss, p, create_graph = True )[0]

                        # print("grad_requires_grad.shape = ", grad_requires_grad.shape)
                        # print("grad_projected.shape = ", grad_projected.shape)
                        # print("prev_d.shape = ", prev_d.shape if prev_d is not None else None)
                        # print("grad_requires_grad.requires_grad = ", grad_requires_grad.requires_grad)
                        # print("grad_projected.requires_grad = ", grad_projected.requires_grad)
                        # print("grad_requires_grad.grad_fn = ", grad_requires_grad.grad_fn)
                        # print("prev_d.requires_grad = ", prev_d.requires_grad if prev_d is not None else None)
                        # input("Press Enter to continue...")

                        grad_times_preconditioned_g = (grad_projected * grad_requires_grad).sum()
                        grad_times_mom = (prev_d * grad_requires_grad).sum()
                        # print(grad_times_preconditioned_g.requires_grad)
                        # print(grad_times_preconditioned_g.grad_fn)
                        # print(grad_times_mom.grad_fn)
                        # input("Press Enter to continue...")

                        Hessian_preconditioned_g = torch.autograd.grad(grad_times_preconditioned_g, p , create_graph = True , retain_graph=True)[0]
                        Hessian_mom = torch.autograd.grad(grad_times_mom, p , create_graph = True, retain_graph=True)[0]

                    preconditioned_g_Hessian_preconditioned_g = (grad_projected * Hessian_preconditioned_g).sum()

                    # print(f"preconditioned_g_Hessian_preconditioned_g.shape = {preconditioned_g_Hessian_preconditioned_g}")

                    mom_Hessian_mom = (prev_d * Hessian_mom).sum()

                    # print(f"mom_Hessian_mom.shape = {mom_Hessian_mom}")

                    preconditioned_g_Hessian_mom = (grad_projected * Hessian_mom).sum()
                    preconditioned_g_Hessian_mom1 = torch.tensordot(
                        grad_projected,
                        Hessian_mom,
                        dims=[[0], [0]], 
                    )

                    if preconditioned_g_Hessian_mom1.dim() > 0:
                        preconditioned_g_Hessian_mom1 = preconditioned_g_Hessian_mom1.trace()

                    # print(f"preconditioned_g_Hessian_mom.shape = {preconditioned_g_Hessian_mom}, preconditioned_g_Hessian_mom1.shape = {preconditioned_g_Hessian_mom1}")
                    # input("Press Enter to continue...")

                    # print(f"preconditioned_g_Hessian_mom.shape = {preconditioned_g_Hessian_mom}")
                    
                    subspace_hessian = torch.tensor([[preconditioned_g_Hessian_preconditioned_g , -preconditioned_g_Hessian_mom], [-preconditioned_g_Hessian_mom, mom_Hessian_mom]], device=grad_projected.device)

                    # print(f"subspace_hessian.shape = {subspace_hessian.shape}, subspace_hessian = {subspace_hessian}")
                    # input("Press Enter to continue...")
                    #if subspace_hessian is not invertible:
                    if not torch.linalg.matrix_rank(subspace_hessian) == subspace_hessian.shape[0]:
                        # print("Subspace Hessian is not invertible, using negative gradient as the initial momentum direction.")
                        # input("Press Enter to continue...")
                        # print("grad_projected" , grad_projected)
                        # print("prev_d" , prev_d)
                        # print("Hessian_preconditioned_g" , Hessian_preconditioned_g)
                        # print("Hessian_mom" , Hessian_mom)
                        # input("Press Enter to continue...")
                        d = -grad.detach().clone() * 1e-4
                    else:
                        c1 = torch.tensordot(mom_projected, grad_projected, dims=[[0], [0]])
                        if c1.dim() > 0:
                            c1 = c1.trace()

                        c = torch.tensor([[-grad_projected.norm('fro') ** 2 ], [c1]], device=grad_projected.device) #Q_T_flat_grad

                        alpha = torch.linalg.solve(subspace_hessian, -c) # alpha = (Hessian + lambda * I)^-1 * Q_T_flat_grad
                        d = -alpha[0] * grad_projected + alpha[1] * prev_d # d = alpha_1 * Q_T_flat_grad + alpha_2 * prev_d
                else:
                    # print("prev_d is None, using negative gradient as the initial momentum direction.")
                    # If no previous momentum direction, use the negative gradient as the initial momentum direction.
                    d = -grad.detach().clone() * 1e-4

                state["step"] += 1
                state["prev_d"] = d.detach().clone() # Save the current momentum direction for the next step.
                
                # print("d.shape = ", d.shape)
                # Update is done after the gradient step to avoid using current gradients in the projection.
                self.update_preconditioner(grad, state, 
                                               max_precond_dim=group['max_precond_dim'],
                                               merge_dims=group["merge_dims"],
                                               precondition_1d=group["precondition_1d"])
        
        for group in self.param_groups:
            for p in group["params"]:
                # print(f"Finally! Updating parameter {p.shape} with grad {p.grad.shape} and state {self.state[p].keys()}")
                # input("Press Enter to continue...")
                state = self.state[p]
                step_size = group["lr"]
                p.add_(state["prev_d"], alpha=step_size) # Update the parameter with the computed direction.


        return loss
    
    def init_preconditioner(self, grad, state, precondition_frequency=10, 
                            shampoo_beta=0.95, max_precond_dim=10000, precondition_1d=False,
                            merge_dims=False):
        """
        Initializes the preconditioner matrices (L and R in the paper).
        """
        state['GG'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state['GG'].append([])
                else:
                    state['GG'].append(torch.zeros(sh, sh, device=grad.device))
                    
        state['Q'] = None # Will hold all the eigenbases of the preconditioner.
        state['precondition_frequency'] = precondition_frequency
        state['shampoo_beta'] = shampoo_beta          
        
    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == 'channels_last':
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [0]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad
        
    def update_preconditioner(self, grad, state, 
                              max_precond_dim=10000, merge_dims=False, precondition_1d=False):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        # if state["Q"] is not None:
        #     state["exp_avg"] = self.project_back(state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim)
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state['GG'][0].lerp_(grad.unsqueeze(1) @ grad.unsqueeze(0), 1-state['shampoo_beta'])
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                                new_grad,
                                new_grad,
                                dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                            )
                        state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                                grad,
                                grad,
                                # Contracts across all dimensions except for k.
                                dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                            )
                        state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
                     
        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])
        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)
            # state['Q'] = self.get_fast_QR(state, max_precond_dim, merge_dims)             

        # if state["step"] > 0:
        #     state["exp_avg"] = self.project(state["exp_avg"], state, merge_dims=merge_dims, max_precond_dim=max_precond_dim) 

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """
        Projects the gradient back to the original space.
        """
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == 'channels_last' and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state['Q']:
            if len(mat) > 0:
                grad = torch.tensordot(
                        grad,
                        mat,
                        dims=[[0], [1]],
                    )
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
                
        if merge_dims:
            if self._data_format == 'channels_last' and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad
        

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)
        
        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0], device=m.device))
            except:
                _, Q = torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final
        

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration 
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state['GG']
        orth_list = state['Q']

        matrix = []
        orth_matrix = []
        for m,o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
        
        # orig_shape = state['exp_avg_sq'].shape
        # if self._data_format == 'channels_last' and len(orig_shape) == 4:
        #     permuted_shape = state['exp_avg_sq'].permute(0, 3, 1, 2).shape
        # if merge_dims:
        #     exp_avg_sq = self.merge_dims(state['exp_avg_sq'], max_precond_dim)
        # else:
        #     exp_avg_sq = state['exp_avg_sq']
            
        final = []
        for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
            if len(m)==0:
                final.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            # exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:,sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        
        # if merge_dims:
        #     if self._data_format == 'channels_last' and len(orig_shape) == 4:
        #         exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
        #     else:
        #         exp_avg_sq = exp_avg_sq.reshape(orig_shape)
                
        # state['exp_avg_sq'] = exp_avg_sq
        return final
    
    