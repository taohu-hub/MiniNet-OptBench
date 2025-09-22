# mypy: allow-untyped-defs
from typing import Optional, Iterable

import torch
#from torch.optimizer import Optimizer, ParamsT
from torch.optim.optimizer import Optimizer  # 修改路径
import torch.nn as nn

ParamsT = Iterable[nn.Parameter]

__all__ = ["LBFGS"]

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
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1,
        max_iter: int = 1,
        max_eval: Optional[int] = 50,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
        verbose: bool = False,
    ):
        self.angle_log = []
        self.test_log = []
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)
        self.verbose = verbose
        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

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
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        break_reason = None  # Initialize the break reason variable
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1
            if state["n_iter"] > 1:
                prev_d = d.clone(memory_format=torch.contiguous_format)
                prev_t = t
            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr
            #d_temp = d.clone(memory_format=torch.contiguous_format)
            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break_reason = "Directional derivative below tolerance."
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn == "strong_wolfe":
                    if state["n_iter"] > 1:
                        mom = prev_t * prev_d
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)
        
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )

                    self._add_grad(t, d)
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                elif line_search_fn == "ps_grad":
                    x_init = self._clone_param()
                    if state["n_iter"] == 1:
                        #print(f'norm of d={norm_d:.4e},norm of mom={norm_m:.4e}')
                        #d = d / norm_d
                        #gtd = gtd / norm_d
                        final_d = d.clone(memory_format=torch.contiguous_format)
                        final_gtd = gtd
                        d = final_d
                    else:
                        mom = prev_t * prev_d
                        #new_d = d.clone(memory_format=torch.contiguous_format)
                        new_d = - flat_grad
                        if torch.norm(mom) > 1e-6:
                            mom = mom/ torch.norm(mom)
                            new_d = new_d / torch.norm(new_d)

                            Q = torch.stack((new_d, mom), dim=1)
                            # compute the projection matrix
                            # compute the hession vector product Hessian of f times new_d and mom respectively
                            # Hessian of f times new_d= gradient at x+eps*new_d - gradient at x / eps
                            # Hessian of f times mom= gradient at x+eps*mom - gradient at x / eps
                            # compute the gradient at x
                            grad_x = self._gather_flat_grad()
                            # compute the gradient at x+eps*new_d
                            grad_new_d = self._directional_evaluate(closure, x_init, 1e-4, new_d)[1]
                            # compute the gradient at x+eps*mom
                            grad_mom = self._directional_evaluate(closure, x_init, 1e-4, mom)[1]
                            # compute the Hessian vector product
                            Hessian_new_d = (grad_new_d - grad_x) / 1e-4            
                            Hessian_mom = (grad_mom - grad_x) / 1e-4
                            # subspace hessian = Q^T * Hessian * Q = Q^T * (Hessian_new_d, Hessian_mom) 
                            subspace_hessian = Q.T @ torch.stack((Hessian_new_d, Hessian_mom), dim=1)
                            # compute eigenvalues of subspace hessian, which is 2 by 2 and easy to compute
                            eigenvalues, eigenvectors = torch.linalg.eig(subspace_hessian)
                            #print(f"Iter={state['n_iter']}, subspace hessian={subspace_hessian}, eigenvalues={eigenvalues}")
                            # final direction equals the inverse of subspace hessian times the gradient Q^T * grad
                            final_d = - Q @ torch.inverse(subspace_hessian) @ Q.T @ flat_grad
                            final_d = final_d / torch.norm(final_d) * torch.norm(new_d)

                            d = final_d.clone(memory_format=torch.contiguous_format)
                                
                            #d = d / torch.norm(d) 
                            final_gtd = flat_grad.dot(final_d)
                            # store the real part of the eigenvalues
                            eigenvalues = torch.real(eigenvalues)

                            def obj_func(x, t, d):
                                return self._directional_evaluate(closure, x, t, d)
                            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                                    obj_func, x_init, t, final_d, loss, flat_grad, final_gtd
                            )
                        else:
                            #print(f"Iter={state['n_iter']}, mom={torch.norm(mom)}")
                            final_d = d.clone(memory_format=torch.contiguous_format)
                            final_gtd = gtd
                            d = final_d
                            def obj_func(x, t, d):
                                return self._directional_evaluate(closure, x, t, d)
                            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                                    obj_func, x_init, t, final_d, loss, flat_grad, final_gtd
                            )
                        #print(f"Iter={state['n_iter']}, final_d_norm={torch.norm(final_d)}, t={t}")

                    if state["n_iter"] > 1 and state["n_iter"] == 1:
                        loss_test, flat_grad_test, t_test, ls_func_evals_test = _strong_wolfe(
                            obj_func, x_init, t, new_d, prev_loss, flat_grad, gtd
                        )
                        loss_test2, flat_grad_test2, t_test2, ls_func_evals_test2 = _strong_wolfe(
                            obj_func, x_init, t, -flat_grad, prev_loss, flat_grad, -flat_grad.dot(flat_grad)
                        )
                        self.test_log.append((prev_loss, loss, loss_test, loss_test2, t, t_test, t_test2))
                    self._add_grad(t, final_d)
                    opt_cond = flat_grad.abs().max() <= tolerance_grad

                elif line_search_fn == "ps_newton":
                    x_init = self._clone_param()
                    if state["n_iter"] == 1:
                        #print(f'norm of d={norm_d:.4e},norm of mom={norm_m:.4e}')
                        #d = d / norm_d
                        #gtd = gtd / norm_d
                        final_d = d.clone(memory_format=torch.contiguous_format)
                        final_gtd = gtd
                        d = final_d
                    else:
                        mom = prev_t * prev_d
                        new_d = d.clone(memory_format=torch.contiguous_format)
                        if torch.norm(mom) > 1e-6:
                            mom = mom/ torch.norm(mom)
                            new_d = new_d / torch.norm(new_d)
                            # Q is a n by 2 matrix, (new_d, mom)
                            Q = torch.stack((new_d, mom), dim=1)
                            # compute the projection matrix
                            # compute the hession vector product Hessian of f times new_d and mom respectively
                            # Hessian of f times new_d= gradient at x+eps*new_d - gradient at x / eps
                            # Hessian of f times mom= gradient at x+eps*mom - gradient at x / eps
                            # compute the gradient at x
                            grad_x = self._gather_flat_grad()
                            # compute the gradient at x+eps*new_d
                            grad_new_d = self._directional_evaluate(closure, x_init, 1e-4, new_d)[1]
                            # compute the gradient at x+eps*mom
                            grad_mom = self._directional_evaluate(closure, x_init, 1e-4, mom)[1]
                            # compute the Hessian vector product
                            Hessian_new_d = (grad_new_d - grad_x) / 1e-4            
                            Hessian_mom = (grad_mom - grad_x) / 1e-4
                            # subspace hessian = Q^T * Hessian * Q = Q^T * (Hessian_new_d, Hessian_mom) 
                            subspace_hessian = Q.T @ torch.stack((Hessian_new_d, Hessian_mom), dim=1)
                            # compute eigenvalues of subspace hessian, which is 2 by 2 and easy to compute
                            eigenvalues, eigenvectors = torch.linalg.eig(subspace_hessian)
                            #print(f"Iter={state['n_iter']}, subspace hessian={subspace_hessian}, eigenvalues={eigenvalues}")
                            # final direction equals the inverse of subspace hessian times the gradient Q^T * grad
                            final_d = - Q @ torch.inverse(subspace_hessian) @ Q.T @ flat_grad
                            final_d = final_d / torch.norm(final_d) * torch.norm(new_d)

                            d = final_d.clone(memory_format=torch.contiguous_format)
                                
                            #d = d / torch.norm(d) 
                            final_gtd = flat_grad.dot(final_d)
                            # store the real part of the eigenvalues
                            eigenvalues = torch.real(eigenvalues)

                            def obj_func(x, t, d):
                                return self._directional_evaluate(closure, x, t, d)
                            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                                    obj_func, x_init, t, final_d, loss, flat_grad, final_gtd
                            )
                        else:
                            #print(f"Iter={state['n_iter']}, mom={torch.norm(mom)}")
                            final_d = d.clone(memory_format=torch.contiguous_format)
                            final_gtd = gtd
                            d = final_d
                            def obj_func(x, t, d):
                                return self._directional_evaluate(closure, x, t, d)
                            loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                                    obj_func, x_init, t, final_d, loss, flat_grad, final_gtd
                            )
                        #print(f"Iter={state['n_iter']}, final_d_norm={torch.norm(final_d)}, t={t}")

                    if state["n_iter"] > 1 and state["n_iter"] == 1:
                        loss_test, flat_grad_test, t_test, ls_func_evals_test = _strong_wolfe(
                            obj_func, x_init, t, new_d, prev_loss, flat_grad, gtd
                        )
                        loss_test2, flat_grad_test2, t_test2, ls_func_evals_test2 = _strong_wolfe(
                            obj_func, x_init, t, -flat_grad, prev_loss, flat_grad, -flat_grad.dot(flat_grad)
                        )
                        self.test_log.append((prev_loss, loss, loss_test, loss_test2, t, t_test, t_test2))
                    self._add_grad(t, final_d)
                    opt_cond = flat_grad.abs().max() <= tolerance_grad

                else:
                    raise RuntimeError(f"line search type {line_search_fn} is not supported")
                #print(f'norm of d={torch.norm(d)},norm of mom={torch.norm(mom)},t*d={t*torch.norm(d)}')
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################]
            gradnorm = torch.norm(flat_grad)
            #print(f'gradient norm={gradnorm}')
            # optimal condition
            if opt_cond:
                break_reason = f"OPTIMAL"
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break_reason = "SS" # small step
                break

            if abs(loss - prev_loss) < tolerance_change:
                break_reason = f"SC" # small change
                break

            if n_iter == max_iter:
                #break_reason = "Reached maximum iterations."
                break

            if current_evals >= max_eval:
                break_reason = "MAXEVAL"
                break
        #print(f'Iter:{state['n_iter']}, t={t}')

        if break_reason is not None and self.verbose:
            print(f"Iter: {n_iter}, Breaking due to: {break_reason}. Loss change={loss-orig_loss:.6e}. Final grad={gradnorm:.6e}")
        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss