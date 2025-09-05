from typing import Optional, Iterable
import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn

ParamsT = Iterable[nn.Parameter]

__all__ = ["DRSOM"]

class DRSOM_check(Optimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1,
        max_iter: int = 1,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        eps : float = 1e-3,
        line_search_fn: Optional[str] = None,
        verbose: bool = False,
    ):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
            eps = eps,
        )
        super().__init__(params, defaults)
        self.verbose = verbose

        if len(self.param_groups) != 1:
            raise ValueError("DRSOM only supports a single parameter group")

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        with torch.enable_grad():
            flat_grad = self._gather_flat_grad().requires_grad_()
        self._set_param(x)
        return loss, flat_grad

    def _gather_generalized_flat_grad(self, loss , retain_the_computation_graph = True):
        # print("We are now gathering flat grad")
        # print(f"The loss is useable{loss.grad_fn}")

        params = [p for g in self.param_groups for p in g['params']]
        # print(params)
        # input("Press Enter to Continue")

        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused = True)
        # print("We have already calculated the grads.")

        views_grad = []
        for p, g in zip(params, grads):
          # print(p)
          # input("Press Enter to Continue")
          # input("Press Enter to Continue")
          # print(g)
          if g is None:
            # print(p.shape)
            # input("Press Enter to Continue")
            view_grad = p.new_zeros(p.numel(), requires_grad=True)
            assert view_grad.require_grad == True, "view_grad has no gradient? Impossible!"
          else:
            view_grad = g.contiguous().view(-1)
          # print(f"view_grad.shape{view_grad.shape}")
          views_grad.append(view_grad)

        # print("So far so good! Every stuffs are collected!")

        return torch.cat(views_grad, 0)

    def step(self, closure):
        """Performs a single optimization step."""
        assert len(self.param_groups) == 1
        closure = torch.enable_grad()(closure)
        # for p in self._params:
        #     print(p)
        #     input("Press Enter to Continue")
        group = self.param_groups[0]
        lr = group["lr"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        eps = group["eps"]

        # Get or initialize state
        state = self.state[self._params[0]]
        state.setdefault("xinit", self._clone_param())
        state.setdefault("n_iter", 0)  # Initialize or retrieve iteration counter
        prev_d = state.get("prev_d", None)
        prev_t = state.get("prev_t", 0)
        state.setdefault("eig_min_FD", [])
        state.setdefault("eig_max_FD", [])
        state.setdefault("eig_cond_FD", [])
        state.setdefault("eig_min_AG", [])
        state.setdefault("eig_max_AG", [])
        state.setdefault("eig_cond_AG", [])
        state.setdefault("gtd", [])
        state.setdefault("loss_change", [])
            # Reuse or calculate flat_grad

        if "orig_loss" in state:
            orig_loss = state["orig_loss"]
        else:
            orig_loss = closure()
        if "flat_grad" in state:
            flat_grad = state["flat_grad"]
        else:
            flat_grad = self._gather_generalized_flat_grad(orig_loss)
        state["orig_loss"] = orig_loss
        state["flat_grad"] = flat_grad
        # Check for optimality
        if flat_grad.abs().max() <= tolerance_grad:
            return state["orig_loss"]

        with torch.no_grad():
            start_iter = 0
            if state["n_iter"] <= start_iter:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr  # Initial step size

        #if prev_d is None or prev_t == 0:
        if state['n_iter'] <= start_iter:
            d = -flat_grad.clone()  # Initialize direction as negative gradient
            d = d.detach().clone()
            # print(f"Initial loss: {orig_loss}, Initial step size: {torch.norm(t * d):.3e}")
        else:
            if state["n_iter"] == start_iter + 1:
                # momentum=current params - state["xinit"]
                mom_temp = []
                for p, xinit in zip(self._params, state["xinit"]):
                    mom_temp.append((p - xinit).view(-1))  # Subtract tensors and flatten
                mom_temp = torch.cat(mom_temp)  # Concatenate all flattened tensors into one
                # print(f"Initial momentum: {torch.norm(mom_temp):.3e}")
                # print(f"checkpoint 1: momentum difference {torch.norm(mom_temp - prev_t * prev_d)}")

                mom = prev_t * prev_d
            else:
                mom = prev_t * prev_d
                mom_temp = prev_t * prev_d
            
            g = -flat_grad
            g = g.detach().clone()
            
            mom = mom / torch.norm(mom)
            mom = mom.detach().clone()

            # print(g)
            # print(mom)
            # input("Press Enter to Continue")

            assert mom.requires_grad == False, "MOMENTUM will interfere with the differentiation process"
            assert g.requires_grad == False, "SHAMPOOGRAD will interfere with the differentiation process"
            assert flat_grad.requires_grad == True, "We can perform derivatives"
            
            Q = torch.stack((g , mom) , dim = 1)
            # print("Q is now defined")
            QT_flat_grad = Q.T @ flat_grad  
            flat_grad_g = torch.dot(flat_grad, g)
            flat_grad_mom = torch.dot(flat_grad, mom)
            Hessian_g_AG = self._gather_generalized_flat_grad(flat_grad_g, retain_the_computation_graph=True)
            Hessian_mom_AG = self._gather_generalized_flat_grad(flat_grad_mom, retain_the_computation_graph=True)

            with torch.no_grad():
                subspace_hessian_AG = Q.T @ torch.stack((Hessian_g_AG, Hessian_mom_AG), dim=1) + eps * torch.eye(2 , device=Q.device, dtype=Q.dtype)
                eigenvalues_AG, _ = torch.linalg.eigh(subspace_hessian_AG)

                if self.verbose == True:
                    print(eigenvalues_AG)

                hessian_cond = abs(eigenvalues_AG.max().item() / eigenvalues_AG.min().item())           # 2 x 1
                
                inv_subspace_hessian_QT_flat_grad_AG = torch.linalg.solve(subspace_hessian_AG, QT_flat_grad)  # 2 x 1
            # print(f"checkpoint 5.4: inv_subspace_hessian difference = {torch.norm(inv_subspace_hessian_QT_flat_grad - inv_subspace_hessian_QT_flat_grad_temp)}")
            d = - Q @ inv_subspace_hessian_QT_flat_grad_AG  # params x 1
            # print(f"checkpoint 6: final step difference {torch.norm(d - d_temp)}, d norm: {torch.norm(d)}, d_temp norm: {torch.norm(d_temp)}")
            #"""
            d = d.detach().clone()
            
        # Compute directional derivative
        gtd = flat_grad.dot(d)
        # print("Directional Derivative is computed.")

        x_init = self._clone_param()
        with torch.no_grad():
            if line_search_fn == "strong_wolfe":
                def obj_func(x, t, d):
                    return self._directional_evaluate(closure, x, t, d)
                with torch.enable_grad():
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(obj_func, x_init, t, d, state["orig_loss"], flat_grad, gtd)
                # if t == 0:
                #     print(f"At iteration {state['n_iter']}, Line search failed to find a suitable step size")
                self._add_grad(t, d)
            else:
                self._add_grad(t, d)
                with torch.enable_grad():
                    loss = closure()
                    flat_grad = self._gather_generalized_flat_grad(loss)

        Printtime = 1
        # Update state with direction, step size, gradient, and iteration count
        state["prev_d"] = d
        state["prev_t"] = t
        state["flat_grad"] = flat_grad
        state["orig_loss"] = loss
        state["n_iter"] += 1  # Increment iteration counter

        return loss

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)
    if x1 == x2:
        return (xmin_bound + xmax_bound) / 2.0
    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1
    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format) #TODO: check if this is needed to add requires_grad_()
            #bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format).requires_grad_()  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals