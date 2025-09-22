import torch
# torch.set_default_dtype(torch.float64)
import torch.distributed as dist
import math


def zeropower_via_newtonschulz5(G, steps: int , eps = 1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if G.ndim < 2:
        return G / (G.norm() + 1e-7)
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    # X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    if X.norm() > 1.3:
        print(X.norm())
        print(G)
        input("Press Enter to Continue")
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True , eps = 1e-7):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.reshape(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps , eps = eps)
    if grad.ndim >= 2:
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    else:
        update *= grad.size(-1)**0.5
    return update

def zero_power_muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True , eps = 1e-7):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.reshape(update.shape[0], -1)
    update_zero = zeropower_via_newtonschulz5(update, steps=ns_steps , eps = eps)
    return update_zero, update

def special_muon_update(drsom_grad, beta=0.95, ns_steps=5, nesterov=True , eps = 1e-7):
    update = zeropower_via_newtonschulz5(drsom_grad, steps=ns_steps , eps = eps)
    if drsom_grad.ndim >= 2:
        update *= max(1, drsom_grad.size(-2) / drsom_grad.size(-1))**0.5
    else:
        update *= drsom_grad.size(-1)**0.5
    return update

def grad_norm(g: torch.Tensor) -> torch.Tensor:
    if g is None:
        return torch.zeros((), device='cpu')
    if g.ndim <= 1:
        # Frobenius for vectors == L2
        return torch.linalg.vector_norm(g)
    if g.ndim > 2:
        # e.g., conv weights: (out_c, in_c, kH, kW) -> (out_c, in_c*kH*kW)
        g = g.reshape(g.shape[0], -1)
    return torch.linalg.matrix_norm(g, ord='nuc')

# --- Rank / SVD diagnostics -------------------------------------------------
@torch.no_grad()
def print_rank(G: torch.Tensor,
               thresholds=(1e-2, 1e-3, 1e-4),
               top_k: int = 20) -> None:
    """
    Print effective rank diagnostics for a gradient tensor.

    - If G is 1D (vector), ranks are trivial so we do not print them.
    - For tensors with ndim > 2 (e.g., conv filters), we reshape to 2D as
      (out_features, in_features) following the Muon convention used elsewhere.
    - We report:
        * numeric ranks at relative thresholds (w.r.t. largest singular value)
        * stable rank: ||G||_F^2 / ||G||_2^2
        * entropy (participation) effective rank based on squared singulars
        * spectral / Frobenius / nuclear norms
        * top singular values (up to top_k)
    """
    import math
    if G is None:
        print("[print_rank] Gradient is None.")
        return

    # If vector, skip ranks
    if G.ndim <= 1:
        print(f"[print_rank] grad shape={tuple(G.shape)} (vector); ranks are trivial, skipping.")
        return

    # Reshape higher-order tensors to 2D matrix as elsewhere in Muon
    if G.ndim > 2:
        M = G.reshape(G.shape[0], -1)
    else:
        M = G

    # Work in float64 for numerical robustness
    M = M.detach().to(dtype=torch.float64)

    # Compute singular values
    try:
        s = torch.linalg.svdvals(M)
    except RuntimeError:
        # Fallback: add tiny jitter in case of numerical issues
        s = torch.linalg.svdvals(M + 1e-12 * torch.randn_like(M))

    if s.numel() == 0:
        print("[print_rank] No singular values (empty matrix).")
        return

    # Basic norms
    s_max = float(s.max().item())
    frob_sq = float(torch.sum(s**2).item())
    frob = frob_sq ** 0.5
    nuc = float(torch.sum(s).item())

    # Effective ranks
    stable_rank = frob_sq / (s_max ** 2 + 1e-12)
    p = (s**2) / (torch.sum(s**2) + 1e-12)
    entropy = float(-(p * (p + 1e-12).log()).sum().item())
    entropy_rank = math.exp(entropy)

    # Numeric ranks at relative thresholds
    rank_at = {f"rank@>{thr:g}": int(torch.sum(s >= thr * s_max).item()) for thr in thresholds}

    # Reporting
    m, n = M.shape
    print("[print_rank] grad matrix shape=", (m, n),
          ", dtype=", M.dtype,
          ", device=", M.device,
          sep="")
    print(f"  spectral(norm-2)={s_max:.6g} | frobenius={frob:.6g} | nuclear={nuc:.6g}")
    print(f"  stable_rank={stable_rank:.6g} | entropy_rank={entropy_rank:.6g}")
    print("  numeric_ranks:", rank_at)

    k = min(top_k, s.numel())
    s_sorted, _ = torch.sort(s, descending=True)
    top_vals = s_sorted[:k].cpu().numpy().tolist()
    if s.numel() > k:
        tail_info = f"... (showing top {k} of {s.numel()})"
    else:
        tail_info = ""
    print("  top singular values:", top_vals, tail_info)

    input("Press Enter to Continue")

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95 , eps = 1e-7):
        # print("initializing muon")
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum , eps = eps)
        # print("default parameters are collected")
        params = list(params)
        # print(params)
        assert isinstance(params, list) , "params are not instances"
        assert len(params) >= 1 , "params is too short"
        assert isinstance(params[0], torch.nn.Parameter), "params[0] is not an instance"
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)
        # print("initialization finished")

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                    print(f"group{group['weight_decay']}")
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

STEP = 'step'
MOMENTUM = 'momentum'
PRECONDITIONER = 'preconditioner'
GRAFT = 'graft'
MUONGRAD = 'muongrad'
STEPSIZE = 'step_size'
DRSOMGRAD = 'drsomgrad'

# Shared mixin for single-device Muon optimizers
class _MuonSharedOps:
    """
    Mixin that centralizes utility helpers used by the single‑device Muon optimizers.
    This removes code duplication across classes.
    """

    def _gather_flat_grad(self, loss, retain_the_computation_graph=True):
        params = [p for g in self.param_groups for p in g['params']]
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=retain_the_computation_graph,
        )

        views_grad = []
        for p, g in zip(params, grads):
            if g is None:
                view_grad = p.new_zeros(p.numel(), requires_grad=True)
            else:
                view_grad = g.contiguous().view(-1)
            views_grad.append(view_grad)

        return torch.cat(views_grad, 0)

    def _gather_muon_grad_and_momentum(self):
        views_muon_grad = []
        views_mom = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                muon_grad = state.get(MUONGRAD)
                momentum = state.get(MOMENTUM)

                if muon_grad is None:
                    views_muon_grad.append(p.new(p.numel()).zero_())
                else:
                    views_muon_grad.append(muon_grad.view(-1))

                if momentum is None:
                    views_mom.append(p.new(p.numel()).zero_())
                else:
                    views_mom.append(momentum.view(-1))

        return torch.cat(views_muon_grad, 0), torch.cat(views_mom, 0)
    
    @torch.no_grad()
    def _gather_flat_momentum(self):
        views_mom = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if MOMENTUM in state:
                    momentum = state.get(MOMENTUM)
                else:
                    momentum = torch.zeros_like(p)

                if momentum is None:
                    views_mom.append(p.new(p.numel()).zero_())
                else:
                    views_mom.append(momentum.view(-1))

        return torch.cat(views_mom, 0)

    @torch.no_grad()
    def _add_grad(self, step_size, update ,lr =1,  weight_decay = 0.0):
      offset = 0
      # print(f"update requires grad{update}")
      # print("We are performing the _add_grad function")
      for group in self.param_groups:
        with torch.no_grad(): 
          for p in group['params']:
            state = self.state[p]
            numel = p.numel()
            # print("numel = p.numel()")

            update_p = update[offset : offset + numel].view_as(p)
            state[MOMENTUM] = update_p
            # print(f"state[STEP]{state[STEP]}")
            # print(f"state[MOMENTUM]{state[MOMENTUM]}")

            p.add_(update_p, alpha=step_size)
            # print("p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)")
            p.mul_(1 - lr * weight_decay)
            offset += numel
            # print(f"{numel} of the elements are added.")
            state[STEP] = state[STEP] + 1

    @torch.no_grad()
    def update_drsom_steps(self, step_size, update):
        offset = 0
        for group in self.param_groups:
            with torch.no_grad(): 
                for p in group['params']:
                    state = self.state[p]
                    numel = p.numel()
                    update_p = update[offset : offset + numel].view_as(p)
                    state[DRSOMGRAD] = update_p


# Shared base class for all single-device Muon optimizers (non-distributed)
class _BaseSingleDeviceMuon(_MuonSharedOps, torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, mul=1.0, eps=1e-7):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, mul=mul, eps=eps)
        super().__init__(params, defaults)

class SingleDeviceMuon(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # print("loss created")

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                # print(f"p.grad is created, it's grad shape is{p.grad.shape}")
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state[STEP] = 0
                    state[MOMENTUM] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
                # print("p has been modified successfully")

        return loss
    
class SingleDeviceMuon_graft_DRSOM_check(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # print("loss created")

        flag = True

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                # print(f"p.grad is created, it's grad shape is{p.grad.shape}")
                state = self.state[p]
                if len(state) == 0:
                    Flag = False
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state[MOMENTUM] = torch.zeros_like(p)
                    state[STEP] = 0
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                update = update.to(p.grad.dtype)
                state[MUONGRAD] = update.reshape(p.shape)
                # print("p has been modified successfully")

        with torch.enable_grad():
            flat_grad = self._gather_flat_grad(loss)
        flat_muon_grad, flat_momentum = self._gather_muon_grad_and_momentum()

        assert flat_grad.requires_grad == True, "Differentiating grad would not yield Hessian!"
        assert flat_muon_grad.requires_grad == False, "shampoo grad would interfere the result of Hessian vector product!"
        assert flat_momentum.requires_grad == False, "momentum would interfere the result of Hessian vector product!"

        # Trying to figure out the drsom step and do grafting

        with torch.enable_grad():
            grad_norm_squared = torch.dot(flat_grad , flat_grad)
            grad_times_mom = torch.dot(flat_grad , flat_momentum)

        Hessian_grad = 0.5 * self._gather_flat_grad(grad_norm_squared , retain_the_computation_graph=True)
        Hessian_mom = self._gather_flat_grad(grad_times_mom , retain_the_computation_graph=True)

        grad_Hessian_grad = torch.dot(flat_grad , Hessian_grad)
        grad_Hessian_mom = torch.dot(flat_momentum , Hessian_grad)
        mom_Hessian_mom = torch.dot(flat_momentum , Hessian_mom)
        subspace_hessian = torch.tensor([[grad_Hessian_grad , -grad_Hessian_mom], [-grad_Hessian_mom, mom_Hessian_mom]], device=flat_grad.device)
        c = torch.tensor([[- grad_norm_squared ], [grad_times_mom]], device=flat_grad.device)

        # print("subspace_hessian and c are calculated.")

        with torch.no_grad():
            cond_number = float(torch.linalg.cond(subspace_hessian))
            det_Q = float(torch.linalg.det(subspace_hessian))
            eigenvalues, _ = torch.linalg.eigh(subspace_hessian)
            print(eigenvalues)


        # print(f"condition_number{cond_number}")

        if det_Q != 0 and cond_number < 1e12:
            with torch.no_grad():
                alpha = torch.linalg.solve(subspace_hessian, -c)

            d_drsom = -alpha[0] * flat_grad + alpha[1] * flat_momentum
        else:
            print("Now the matrix is singular, we are now using gradient descent.")
            d_drsom = - flat_grad * group["lr"]
        # print("The direction would be cloned.")
        d_drsom = d_drsom.detach().clone()

        state = self.state[group["params"][0]]

        if STEPSIZE not in state:
            state[STEPSIZE] = group['lr']

        log_step_size = torch.log10(1.0/ flat_muon_grad.norm() * d_drsom.norm())

        state[STEPSIZE] = 10 ** ((math.log10(state[STEPSIZE]) * state[STEP] + log_step_size) / (state[STEP] + 1))

        if state[STEP] % 50 == 0:
            print(f"STEP:{state[STEP]} , STEPSIZE:{state[STEPSIZE]}, loss:{loss}")

        d = - flat_muon_grad * state[STEPSIZE]

        # print("The direction would be added.")
        self._add_grad(1, d)
        # print("We have succeeded after a round!")

        return loss

class SingleDeviceMuon_graft_loss(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # print("loss created")

        flag = True

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                # print(f"p.grad is created, it's grad shape is{p.grad.shape}")
                state = self.state[p]
                if len(state) == 0:
                    Flag = False
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state[MOMENTUM] = torch.zeros_like(p)
                    state[STEP] = 0
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                update = update.to(p.grad.dtype)
                state[MUONGRAD] = update.reshape(p.shape)
                # print("p has been modified successfully")

        flat_muon_grad, flat_momentum = self._gather_muon_grad_and_momentum()

        state = self.state[group["params"][0]]

        if STEPSIZE not in state:
            state[STEPSIZE] = group['lr']

        state[STEPSIZE] = min(loss, group['lr'])
        #10 ** ((math.log10(state[STEPSIZE]) * state[STEP] + log_step_size) / (state[STEP] + 1))

        # print(f"STEP:{state[STEP]} , STEPSIZE:{state[STEPSIZE]}")

        d = - group['mul'] * flat_muon_grad * state[STEPSIZE]

        # print("The direction would be added.")
        self._add_grad(1, d , torch.linalg.norm(d) , group['weight_decay'])
        # print("We have succeeded after a round!")

        return loss

class SingleDeviceMuon_graft_gradient(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # print("loss created")

        flag = True
        grad_nuclear_norm = 0
        tot_muon_rank = 0
        muon_operator_norm = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                
                print_rank(p.grad)

                grad_norm_g = grad_norm(p.grad)
                grad_norm_g = grad_norm_g.item()
                grad_nuclear_norm = grad_nuclear_norm + grad_norm_g
                state = self.state[p]
                if len(state) == 0:
                    Flag = False
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state[MOMENTUM] = torch.zeros_like(p)
                    state[STEP] = 0
                update = zero_power_muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                update = update.to(p.grad.dtype)
                state[MUONGRAD] = update.reshape(p.shape)
                                # Frobenius norm squared of MUON update (works for any tensor shape)

                if STEPSIZE not in state:
                    state[STEPSIZE] = group['lr']
                state[STEPSIZE] = min(grad_norm_g, group['lr'])

                if state[MUONGRAD].ndim >= 2:
                    sigma_max = torch.linalg.matrix_norm(state[MUONGRAD], ord=2)
                else:
                    sigma_max = torch.linalg.norm(state[MUONGRAD])
                muon_fro_squared = (state[MUONGRAD] * state[MUONGRAD]).sum()
                tot_muon_rank = tot_muon_rank + muon_fro_squared.item()      
                muon_operator_norm = muon_operator_norm + sigma_max.item()

                state[MUONGRAD] = state[STEPSIZE] * state[MUONGRAD]
                # print(sigma_max)
                # print(p.grad.shape)
                # input("Press Enter to Continue")
                # print("p has been modified successfully")

                if state[STEP] % 100 == 0:
                    print(f"grad nuclear norm{grad_norm_g} shape = {p.grad.shape}")

        flat_muon_grad, flat_momentum = self._gather_muon_grad_and_momentum()

        state = self.state[group["params"][0]]

        # if state[STEP] % 20 == 0:
        #     print(f"grad_nuclear_norm = {grad_nuclear_norm} loss = {loss}")

        d = - group['mul'] * flat_muon_grad

        # print("The direction would be added.")
        self._add_grad(1, d , group['lr'] , group['weight_decay'])
        # print("We have succeeded after a round!")

        return loss, tot_muon_rank, muon_operator_norm

class SingleDeviceMuon_graft_uniform_gradient(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # print("loss created")

        flag = True
        grad_nuclear_norm = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                # print(f"p.grad is created, it's grad shape is{p.grad.shape}")
                # print_rank(p.grad)
                
                state = self.state[p]
                if len(state) == 0:
                    Flag = False
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state[MOMENTUM] = torch.zeros_like(p)
                    state[STEP] = 0
                update, mom = zero_power_muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                grad_norm_g = (update * mom).sum()
                grad_norm_g = grad_norm_g.item()
                grad_nuclear_norm = max(grad_norm_g, grad_nuclear_norm)
                update = update.to(p.grad.dtype)
                state[MUONGRAD] = update.reshape(p.shape)

        flat_muon_grad, flat_momentum = self._gather_muon_grad_and_momentum()

        state = self.state[group["params"][0]]

        # if state[STEP] % 20 == 0:
        #     print(f"grad_nuclear_norm = {grad_nuclear_norm} loss = {loss}")

        d = - group['mul'] * grad_nuclear_norm * flat_muon_grad

        # print("The direction would be added.")
        self._add_grad(1, d , group['lr'] , group['weight_decay'])
        # print("We have succeeded after a round!")

        return loss


class SingleDeviceMuon_DRSOM_check(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # print("loss created")

        flag = True

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                # print(f"p.grad is created, it's grad shape is{p.grad.shape}")
                state = self.state[p]
                if len(state) == 0:
                    Flag = False
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state[MOMENTUM] = torch.zeros_like(p)
                    state[STEP] = 0
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                update = update.to(p.grad.dtype)
                state[MUONGRAD] = update.reshape(p.shape)
                # print("p has been modified successfully")

        with torch.enable_grad():
            flat_grad = self._gather_flat_grad(loss)
        flat_muon_grad, flat_momentum = self._gather_muon_grad_and_momentum()

        assert flat_grad.requires_grad == True, "Differentiating grad would not yield Hessian!"
        assert flat_muon_grad.requires_grad == False, "shampoo grad would interfere the result of Hessian vector product!"
        assert flat_momentum.requires_grad == False, "momentum would interfere the result of Hessian vector product!"

        # print(flat_grad.dtype)
        # print(flat_muon_grad.dtype)
        with torch.enable_grad():
            grad_times_muon_grad = torch.dot(flat_grad , flat_muon_grad)
            grad_times_mom = torch.dot(flat_grad , flat_momentum)
        # print("successfully dotted")
        # print(f"grad_times_muon_grad{grad_times_muon_grad.grad_fn}")
        # print(f"grad_times_mom{grad_times_mom.grad_fn}")

        Hessian_shampoo_grad = self._gather_flat_grad(grad_times_muon_grad , retain_the_computation_graph=True)
        Hessian_mom = self._gather_flat_grad(grad_times_mom , retain_the_computation_graph=True)

        shampoo_grad_Hessian_shampoo_grad = torch.dot(flat_muon_grad , Hessian_shampoo_grad)
        # print("OK, we can shampoo H shampoo")
        mom_Hessian_mom = torch.dot(flat_momentum , Hessian_mom)
        # print("OK, we can mom_H_mom")
        shampoo_grad_Hessian_mom = torch.dot(flat_momentum , Hessian_shampoo_grad)
        # print("OK, we can shampoo H mom")

        subspace_hessian = torch.tensor([[shampoo_grad_Hessian_shampoo_grad , -shampoo_grad_Hessian_mom], [-shampoo_grad_Hessian_mom, mom_Hessian_mom]], device=flat_grad.device)
        c = torch.tensor([[- grad_times_muon_grad ], [grad_times_mom]], device=flat_grad.device)

        # print("subspace_hessian and c are calculated.")

        with torch.no_grad():
            cond_number = float(torch.linalg.cond(subspace_hessian))
            det_Q = float(torch.linalg.det(subspace_hessian))

        if det_Q != 0 and cond_number < 1e12:
            with torch.no_grad():
                alpha = torch.linalg.solve(subspace_hessian, -c)

            d = -alpha[0] * flat_muon_grad + alpha[1] * flat_momentum
        else:
        # print("Now the matrix is singular, we are now using gradient descent.")
            d = - flat_muon_grad * group["lr"]
        # print("The direction would be cloned.")
        d = d.detach().clone()

        # print("The direction would be added.")
        self._add_grad(1, d)
        # print("We have succeeded after a round!")

        return loss

class SingleDeviceDRSOM_check_Muon(_BaseSingleDeviceMuon):
    """
    Muon variant for usage in non-distributed settings.
    """

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]

        with torch.enable_grad():
            flat_grad = self._gather_flat_grad(loss)
        flat_momentum = self._gather_flat_momentum()

        assert flat_grad.requires_grad == True, "Differentiating grad would not yield Hessian!"
        assert flat_momentum.requires_grad == False, "momentum would interfere the result of Hessian vector product!"

        with torch.enable_grad():
            grad_times_grad = torch.dot(flat_grad , flat_grad)
            grad_times_mom = torch.dot(flat_grad , flat_momentum)

        Hessian_grad = 0.5 * self._gather_flat_grad(grad_times_grad , retain_the_computation_graph=True)
        Hessian_mom = self._gather_flat_grad(grad_times_mom , retain_the_computation_graph=True)

        grad_Hessian_grad = torch.dot(flat_grad , Hessian_grad)
        # print("OK, we can shampoo H shampoo")
        mom_Hessian_mom = torch.dot(flat_momentum , Hessian_mom)
        # print("OK, we can mom_H_mom")
        grad_Hessian_mom = torch.dot(flat_momentum , Hessian_grad)
        # print("OK, we can shampoo H mom")

        subspace_hessian = torch.tensor([[grad_Hessian_grad , -grad_Hessian_mom], [-grad_Hessian_mom, mom_Hessian_mom]], device=flat_grad.device)
        c = torch.tensor([[- grad_times_grad], [grad_times_mom]], device=flat_grad.device)

        # print("subspace_hessian and c are calculated.")

        with torch.no_grad():
            cond_number = float(torch.linalg.cond(subspace_hessian))
            det_Q = float(torch.linalg.det(subspace_hessian))

        if grad_Hessian_grad > 0 and det_Q > 0 and cond_number < 1e12:
            with torch.no_grad():
                alpha = torch.linalg.solve(subspace_hessian, -c)

            d = -alpha[0] * flat_grad + alpha[1] * flat_momentum
        else:
            # if cond_number < 1e12:
            #     print("The subspace_hessian is not guaranteed to be psd!")
            # else:
            #     print("The subspace hessian is psd")
            d = - flat_grad * group["lr"]
        d = d.detach().clone()
        self.update_drsom_steps(1, d)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                assert DRSOMGRAD in state, "We have a miss in DRSOM grad here!"
                if len(state) == 0:
                    state[MOMENTUM] = torch.zeros_like(p)
                    state[STEP] = 0
                update = special_muon_update(state[DRSOMGRAD] , eps = group['eps'])
                update = update.to(p.grad.dtype)
                update = update.reshape(p.shape)
                state[MOMENTUM] = update
                p.add_(update, alpha=group['lr'])
                # print("p has been modified successfully")

        return loss



def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                allowed = {"params","lr","momentum","weight_decay","use_muon","betas","eps"}
                assert set(group.keys()).issubset(allowed)
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                allowed = {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
                assert set(group.keys()).issubset(allowed)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"] , eps = group['eps'])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
