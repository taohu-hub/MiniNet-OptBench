import os
import numpy as np
from collections import deque

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from opts.Subspace.func import generate_sample_points
from opts.Subspace.func import quadratic_regression
from opts.Subspace.func import solve_for_tau

# NOTE: 1 means we need record of information, 0 means not
INFO_OPT = 0

# NOTE: 1 means return gradient norm, 0 means not
GRAD_NORM_OPT = 1

# TODO: it is hard-coded now to avoid numerical issues
epsilon = 1e-6

neta = 0.15

# TODO: it is hard-coded now
sample_radius_0 = 1e-2
sample_radius_opt = 1


# record all the needed information for output
class his_manager:
    def __init__(self):
        self.niter = 0

        self.y_iter_his = []
        self.tr_radius_his = []
        self.sample_radius_his = []

        self.rho_his = []
        self.grad_norm_his = []
        self.step_size_his = []


# subspace solver
class SUBSPACE(torch.optim.Optimizer):
    def __init__(self, params):
        self.his = his_manager()
        defaults = {}
        super(SUBSPACE, self).__init__(params, defaults)

    # step 1
    def init_option(
        self,
        mode="default",
        construct_model_opt=1,
        extra_sample_num=0,
        mom_num=1,
        gd_num=1,
        ds_num=0,
        eig_opt=0,
        prox_newton_opt=0,
        lanczos_freq=50,
        lanczos_step_num=10,
        lbfgs_opt=0,
        history_size=100,
    ):
        self.mode = mode
        self.construct_model_opt = construct_model_opt
        # NOTE: 0 means sampling method, 1 means hvp method
        # TODO: self.construct_model_opt can be adaptively changed during optimization
        if self.construct_model_opt == 0:
            self.extra_sample_num = extra_sample_num

        self.mom_num = mom_num
        self.gd_num = gd_num
        self.ds_num = ds_num

        self.eig_opt = eig_opt
        if self.eig_opt == 1:
            self.prox_newton_opt = prox_newton_opt
            # NOTE: 0 means only negative eigen direction; 1 means both; 2 means only approximated newton direction
            self.lanczos_freq = lanczos_freq
            self.lanczos_step_num = lanczos_step_num

        self.lbfgs_opt = lbfgs_opt
        if self.lbfgs_opt == 1:
            self.history_size = history_size

    # step 2
    def init_hyperparams(
        self,
        tr_radius_0=0.5,
        tr_radius_max=10,
        rou_1_bar=0.25,
        rou_2_bar=0.75,
        gamma_1=0.8,
        gamma_2=2,
    ):
        self.tr_radius_0 = tr_radius_0
        self.tr_radius_max = tr_radius_max
        self.rou_1_bar = rou_1_bar
        self.rou_2_bar = rou_2_bar
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    @torch.no_grad()
    def _get_param(self):
        """get all parameters in all param_groups with gradient requirements"""
        return [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]

    @torch.no_grad()
    def _clone_param(self):
        # a list
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    @torch.no_grad()
    def _set_param(self, params_data):
        # a list
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata.clone(memory_format=torch.contiguous_format))

    @torch.no_grad()
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(p.numel() for p in self._params)
        return self._numel_cache

    @torch.no_grad()
    def _add_d(self, d, step_size=1):
        # 1D tensor
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.add_(d[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def get_value(self, d_new, closure):
        # 1D tensor
        self._add_d(d_new)
        y = closure(backward=False)
        self._set_param(self.params_copy)
        # NOTE: y should be detached when used
        return y

    def flatten(self, ls):
        views = [u.view(-1) for u in ls]
        return torch.cat(views)  # 1D tensor, require grad

    def hv_with_closure(self, v, closure):
        # 1D tensor
        loss = closure(backward=False)
        grad = torch.autograd.grad(loss, self._params, create_graph=True)
        grad_vec = self.flatten(grad)
        gv = torch.dot(grad_vec, v)
        hv_temp = torch.autograd.grad(
            gv, self._params
        )  # it is a list that represents hv
        hv = [u.detach() for u in hv_temp]
        hv_vec = self.flatten(hv)
        return hv_vec

    def hv_with_grad(self, v, grad_vec):
        gv = torch.dot(grad_vec, v)
        hv_temp = torch.autograd.grad(
            gv, self._params, retain_graph=True
        )  # it is a list that represents hv
        # Note: we should retain graph because grad will be used multiple times in lanczos method
        hv = [u.detach() for u in hv_temp]
        hv_vec = self.flatten(hv)
        return hv_vec

    def test_hv(self, closure):
        v = torch.rand(self._numel())
        hv = self.hv_with_closure(v, closure)
        print(hv)

    @torch.no_grad()
    def add_together(self, v_ls, coeff):
        # list of 1D tensor
        coeff = coeff.flatten()  # accept 1D tensor or (n,1) (1,n) tensor
        V = torch.stack(v_ls)
        return torch.matmul(coeff, V)  # 1D tensor

    def lanczos(self, n, grad_vec, mode="default"):
        T = torch.zeros(n, n)
        q = -self.grad_vec
        beta = torch.norm(q)
        v_ls = [torch.zeros_like(q)]
        i = 0
        while True:
            v = q / beta
            v_ls.append(v)
            Hv = self.hv_with_grad(v, grad_vec)
            alpha = torch.dot(v, Hv)
            T[i][i] = alpha
            if i == n - 1:
                break
            else:
                q = Hv - alpha * v - beta * v_ls[i]
                beta = torch.norm(q)
                if beta < epsilon:
                    n = i + 1
                    T = T[:n, :n]
                    break
                else:
                    T[i + 1][i] = beta
                    T[i][i + 1] = beta
                    i += 1

        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        min_eig, min_index = torch.min(eigenvalues, dim=0)

        # print some information about eigenvalue
        # print('Minimal eigenvalue: ', min_eig.cpu().item())

        neg_vtg = torch.zeros(n)
        for i in range(n):
            neg_vtg[i] = -torch.dot(v_ls[i + 1], self.grad_vec)
        y = torch.linalg.solve(T, neg_vtg)
        newton = self.add_together(v_ls[1:], y)

        if mode == "default":
            sign = min_eig < 0
            if sign:
                min_eig_vec = self.add_together(v_ls[1:], eigenvectors[:, min_index])
            else:
                min_eig_vec = None
            return sign, min_eig_vec, newton
        elif mode == "test":
            min_eig_vec = self.add_together(v_ls[1:], eigenvectors[:, min_index])
            return min_eig, min_eig_vec, newton

    def test_lanczos(self, closure):
        self.loss = closure(backward=False)
        self.y_current = self.loss.detach()

        grad = torch.autograd.grad(self.loss, self._params, create_graph=True)
        grad_vec = self.flatten(grad)
        self.grad_vec = grad_vec.detach()

        min_eig, min_eig_vec, newton = self.lanczos(
            self._numel(), grad_vec, mode="test"
        )
        H_v = self.hv_with_closure(min_eig_vec, closure)
        lamda_v = min_eig * min_eig_vec
        print(torch.norm(H_v - lamda_v))
        neg_g = self.hv_with_closure(newton, closure)
        print(torch.norm(-self.grad_vec - neg_g))

    @torch.no_grad()
    def LBFGS(self):
        flat_grad = self.grad_vec.clone(memory_format=torch.contiguous_format)

        old_dirs = self.state.get("old_dirs")
        old_stps = self.state.get("old_stps")
        ro = self.state.get("ro")
        H_diag = self.state.get("H_diag")
        prev_flat_grad = self.state.get("prev_flat_grad")

        if self.his.niter == 0:
            d = flat_grad.neg()
            old_dirs = []
            old_stps = []
            ro = []
            H_diag = 1
        else:
            s = self.d_star.clone(memory_format=torch.contiguous_format)

            # do lbfgs update (update memory)
            y = flat_grad.sub(prev_flat_grad)
            ys = y.dot(s)  # y*s
            # TODO: change 1e-10 to epsilon
            if ys > 1e-10:
                # updating memory
                if len(old_dirs) == self.history_size:
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

            if "al" not in self.state:
                self.state["al"] = [None] * self.history_size
            al = self.state["al"]

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

        self.state["old_dirs"] = old_dirs
        self.state["old_stps"] = old_stps
        self.state["ro"] = ro
        self.state["H_diag"] = H_diag
        self.state["prev_flat_grad"] = prev_flat_grad

        return d

    # step 3
    def start(self):
        if self.mom_num > 0:
            self.mom_buffer = deque(maxlen=self.mom_num)

        if self.gd_num > 0:
            self.gd_buffer = deque(maxlen=self.gd_num)

        if self.eig_opt == 1:
            if self.prox_newton_opt == 0:
                self.eig_buffer = deque(maxlen=1)
            elif self.prox_newton_opt == 1:
                self.eig_buffer = deque(maxlen=1)
                self.prox_newton_buffer = deque(maxlen=1)
            elif self.prox_newton_opt == 2:
                self.prox_newton_buffer = deque(maxlen=1)
            self.lanczos_sign = 1  # 1 means it is time to perform lanczos, 2,3,4...lanczos_freq means not

        if self.lbfgs_opt == 1:
            self.lbfgs_buffer = deque(maxlen=1)

        self._params = self._get_param()
        self._numel_cache = None

        self.tr_radius = self.tr_radius_0

        if self.construct_model_opt == 0:
            self.sample_radius = sample_radius_0

        # TODO: now this hyperparameter is hard-coded
        if self.mode == "non-m":
            self.M = 20

        # NOTE: it is used in LBFGS direction part
        if self.lbfgs_opt == 1:
            self.state = {}

        for param in self._params:
            param.requires_grad = True

        self.device = self._params[0].device

    @torch.no_grad()
    def get_directions(self):
        directions = []
        if self.gd_num > 0:
            directions += list(self.gd_buffer)
        if self.mom_num > 0 and self.his.niter > 0:
            directions += list(self.mom_buffer)
        if self.ds_num > 0:
            for _ in range(self.ds_num):
                directions.append(torch.rand(self._numel()))
        if self.eig_opt == 1 and self.lanczos_sign == 1:
            if self.prox_newton_opt == 0:
                directions += list(self.eig_buffer)
            elif self.prox_newton_opt == 1:
                directions += list(self.eig_buffer)
                directions += list(self.prox_newton_buffer)
            elif self.prox_newton_opt == 2:
                directions += list(self.prox_newton_buffer)
        if self.lbfgs_opt == 1:
            directions += list(self.lbfgs_buffer)
        self.directions = torch.stack(directions)  # each row is a direction

    @torch.no_grad()
    def preprocess_directions(self):
        Q, _ = torch.linalg.qr(self.directions.T, mode="reduced")
        self.basis = Q.T  # each row is a direction
        self.sub_dim = self.basis.shape[0]

        # print(self.sub_dim)

    @torch.no_grad()
    def transform_from_vec_to_d(self, vec):
        # vec is an np.array
        coeff = torch.tensor(vec, device=self.device, dtype=torch.float32).flatten()
        d_new = torch.matmul(coeff, self.basis)
        return d_new

    def construct_model(self, closure):
        self.subspace_const = self.y_current.cpu().item()
        subspace_grad = torch.matmul(self.grad_vec, self.basis.T)
        subspace_grad = subspace_grad.cpu().numpy()

        num = max(1, 0.5 * self.sub_dim * (self.sub_dim + 1))
        num += self.extra_sample_num
        num = int(num)

        self.r = self.sample_radius
        # NOTE: self.r is used in draw landscape
        X = generate_sample_points(self.sub_dim, num, radius=self.r)

        y_ls = [0]
        for n in range(num):
            d_new = self.transform_from_vec_to_d(X[n])
            exact_val = self.get_value(d_new, closure).detach().cpu().item()
            y_ls.append(exact_val - self.subspace_const - np.dot(subspace_grad, X[n]))

        P = np.vstack((np.zeros((1, self.sub_dim)), X))
        y = np.array(y_ls)

        self.subspace_hessian = quadratic_regression(P, y)
        self.subspace_grad = subspace_grad.reshape(-1, 1)

        # NOTE: update sample radius
        if sample_radius_opt == 0:
            pass
        elif sample_radius_opt == 1:
            key = (np.max(y) - np.min(y)) / max(1, abs(self.subspace_const))
            level = 0.5 * 1e-4
            if key < level:
                self.sample_radius *= 1.2
            elif key > level:
                self.sample_radius *= 0.8

    def test_model(self, p, closure):
        d_new = self.transform_from_vec_to_d(p)
        exact_val = self.get_value(d_new, closure).detach().cpu().item()
        model_val = self.get_model_val(p)
        return abs(exact_val - model_val)

    def construct_model_hvp(self, grad):
        self.subspace_const = self.y_current.cpu().item()
        subspace_grad = torch.matmul(self.grad_vec, self.basis.T)
        self.subspace_grad = subspace_grad.cpu().numpy()

        Hd_ls = []
        for d in self.basis:
            Hd = self.hv_with_grad(d, grad)
            Hd_ls.append(Hd)
        Hds = torch.stack(Hd_ls)
        subspace_hessian = torch.matmul(self.basis, Hds.T)

        self.subspace_hessian = subspace_hessian.cpu().numpy()
        self.subspace_grad = self.subspace_grad.reshape(-1, 1)

    @torch.no_grad()
    def get_model_val(self, p):
        val = (
            self.subspace_const
            + p.T @ self.subspace_grad
            + 0.5 * p.T @ self.subspace_hessian @ p
        )
        return val[0][0]

    @torch.no_grad()
    def truncated_CG(self):
        max_iter = self.sub_dim * 100

        s = np.zeros((self.sub_dim, 1))
        r = self.subspace_grad
        r_norm_0 = np.linalg.norm(r)
        p = -self.subspace_grad
        k = 0
        while k < max_iter:
            if p.T @ self.subspace_hessian @ p <= 0:
                t = solve_for_tau(s, p, self.tr_radius)
                return s + t * p
            alpha = (r.T @ r) / (p.T @ self.subspace_hessian @ p)
            s_new = s + alpha * p
            if np.linalg.norm(s_new) >= self.tr_radius:
                t = solve_for_tau(s, p, self.tr_radius)
                return s + t * p
            r_new = r + alpha * (self.subspace_hessian @ p)
            if np.linalg.norm(r_new) < min(epsilon, epsilon * r_norm_0):
                return s_new
            beta = (r_new.T @ r_new) / (r.T @ r)
            p = -1 * r_new + beta * p
            k += 1
            s = s_new
            r = r_new
        return s

    def step(self, closure):
        if self.his.niter == 0:
            # copy params
            self.params_copy = self._clone_param()

            self.loss = closure(backward=False)
            self.y_current = self.loss.detach()
            grad = torch.autograd.grad(self.loss, self._params, create_graph=True)
            grad_vec = self.flatten(grad)
            self.grad_vec = grad_vec.detach()

            # upgrade directions in the buffer
            if self.gd_num > 0:
                self.gd_buffer.append(
                    self.grad_vec.clone(memory_format=torch.contiguous_format)
                )

            if self.eig_opt == 1 and self.lanczos_sign == 1:
                sign, min_eig_vec, prox_newton = self.lanczos(
                    self.lanczos_step_num, grad_vec
                )
                if self.prox_newton_opt == 0:
                    if sign:
                        self.eig_buffer.append(min_eig_vec)
                elif self.prox_newton_opt == 1:
                    if sign:
                        self.eig_buffer.append(min_eig_vec)
                    self.prox_newton_buffer.append(prox_newton)
                elif self.prox_newton_opt == 2:
                    self.prox_newton_buffer.append(prox_newton)

            if self.lbfgs_opt == 1:
                self.lbfgs_buffer.append(
                    self.LBFGS().clone(memory_format=torch.contiguous_format)
                )

            # construct basis
            self.get_directions()
            self.preprocess_directions()

            # construct model
            if self.construct_model_opt == 0:
                for param in self._params:
                    param.requires_grad = False
                self.construct_model(closure)
                for param in self._params:
                    param.requires_grad = True
            elif self.construct_model_opt == 1:
                self.construct_model_hvp(grad_vec)
        elif self.rho > neta:
            # copy params
            self.params_copy = self._clone_param()

            grad = torch.autograd.grad(self.loss, self._params, create_graph=True)
            grad_vec = self.flatten(grad)
            self.grad_vec = grad_vec.detach()

            # upgrade directions in the buffer
            if self.gd_num > 0:
                self.gd_buffer.append(
                    self.grad_vec.clone(memory_format=torch.contiguous_format)
                )

            if self.eig_opt > 0 and self.lanczos_sign == 1:
                sign, min_eig_vec, prox_newton = self.lanczos(
                    self.lanczos_step_num, grad_vec
                )
                if self.prox_newton_opt == 0:
                    if sign:
                        self.eig_buffer.append(min_eig_vec)
                elif self.prox_newton_opt == 1:
                    if sign:
                        self.eig_buffer.append(min_eig_vec)
                    self.prox_newton_buffer.append(prox_newton)
                elif self.prox_newton_opt == 2:
                    self.prox_newton_buffer.append(prox_newton)

            if self.lbfgs_opt == 1:
                self.lbfgs_buffer.append(
                    self.LBFGS().clone(memory_format=torch.contiguous_format)
                )

            # construct basis
            self.get_directions()
            self.preprocess_directions()

            # construct model
            if self.construct_model_opt == 0:
                for param in self._params:
                    param.requires_grad = False
                self.construct_model(closure)
                for param in self._params:
                    param.requires_grad = True
            elif self.construct_model_opt == 1:
                self.construct_model_hvp(grad_vec)
        elif self.ds_num > 0:
            # construct basis
            self.get_directions()
            self.preprocess_directions()

            # construct model
            if self.construct_model_opt == 0:
                for param in self._params:
                    param.requires_grad = False
                self.construct_model(closure)
                for param in self._params:
                    param.requires_grad = True
            elif self.construct_model_opt == 1:
                self.construct_model_hvp(grad_vec)
        elif self.construct_model_opt == 0:
            # construct model
            for param in self._params:
                param.requires_grad = False
            self.construct_model(closure)
            for param in self._params:
                param.requires_grad = True
        else:
            pass

        # record information: current function value and current trust region radius
        if INFO_OPT == 1:
            self.his.y_iter_his.append(self.y_current.cpu().item())
            self.his.tr_radius_his.append(self.tr_radius)
            if self.construct_model_opt == 0:
                self.his.sample_radius_his.append(self.sample_radius)

        # get p_star
        self.p_star = self.truncated_CG()
        self.p_star_norm = np.linalg.norm(self.p_star)

        # get d_star
        self.d_star = self.transform_from_vec_to_d(self.p_star)
        self._add_d(self.d_star)
        loss = closure(backward=False)
        self.tcg_y = loss.detach()

        tcg_y_model = self.get_model_val(self.p_star)
        y_current_model = self.get_model_val(np.zeros((self.sub_dim, 1)))

        if self.mode == "default":
            if y_current_model - tcg_y_model > 0:
                self.rho = (self.y_current.cpu().item() - self.tcg_y.cpu().item()) / (
                    y_current_model - tcg_y_model
                )
            else:
                print("TCG generate a wrong point")
                self.rho = -1
        elif self.mode == "non-m":
            y_max = max(self.his.y_iter_his[-self.M :])
            if y_current_model - tcg_y_model > 0:
                self.rho = (y_max - self.tcg_y.cpu().item()) / (
                    y_current_model - tcg_y_model
                )
            else:
                print("TCG generate a wrong point")
                self.rho = -1

        # update new point
        if self.rho > neta:
            self.loss = loss
            self.y_current = self.tcg_y
        else:
            self._set_param(self.params_copy)

        self.step_end()

        # return the norm of gradient
        if GRAD_NORM_OPT == 1:
            return self.grad_norm

    @torch.no_grad()
    def para_upd(self):
        if self.rho < self.rou_1_bar:
            self.tr_radius *= self.gamma_1
        elif self.rho > self.rou_2_bar and self.p_star_norm >= 0.9 * self.tr_radius:
            self.tr_radius = min(self.tr_radius * self.gamma_2, self.tr_radius_max)

    @torch.no_grad()
    def mom_upd(self):
        self.mom_buffer.append(self.d_star)

    @torch.no_grad()
    def step_end(self):
        # NOTE: the norm of subspace gradient equals to the norm of gradient if gd_num = 1 ( gd becomes the first base)
        if GRAD_NORM_OPT == 1 or INFO_OPT == 1:
            self.grad_norm = torch.norm(self.grad_vec).cpu().item()
        if INFO_OPT == 1:
            if self.rho > neta:
                self.step_size = torch.norm(self.d_star).cpu().item()
            else:
                self.step_size = 0

        # output some relative information for observation
        if INFO_OPT == 1:
            print(
                f"Iteration: {self.his.niter} | Function value: {self.y_current.cpu().item():.6f} | Rho: {self.rho:.4f} | Trust region radius: {self.tr_radius:.6f} | Step size: {self.step_size:.6f} | Gradient norm: {self.grad_norm:.4f}"
            )

        # record information
        self.his.niter += 1
        if INFO_OPT == 1:
            self.his.rho_his.append(self.rho)
            self.his.grad_norm_his.append(self.grad_norm)
            self.his.step_size_his.append(self.step_size)

        # update momentum
        if self.rho > neta:
            if self.mom_num > 0:
                self.mom_upd()
        else:
            pass

        # update hyperparameters
        self.para_upd()

        # update lanczos_sign
        if self.eig_opt == 1:
            if self.lanczos_sign == self.lanczos_freq:
                self.lanczos_sign = 1
            else:
                self.lanczos_sign += 1

    # NOTE: this function should be used before one step
    @torch.no_grad()
    def check_stop(self, tr_radius_tol=1e-6, grad_norm_tol=1e-6):
        if self.tr_radius < tr_radius_tol:
            return 1
        if self.his.niter > 0 and (GRAD_NORM_OPT == 1 or INFO_OPT == 1):
            if self.grad_norm < grad_norm_tol:
                return 1
        return 0

    # NOTE: this function should be used in the end when INFO_OPT == 1
    def draw_info(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        (ax1, ax2), (ax3, ax4) = axes

        ax1.plot(self.his.y_iter_his)
        ax2.plot(self.his.tr_radius_his, label="Trust radius")
        ax2.plot(self.his.step_size_his, label="Step size")
        if self.construct_model_opt == 0:
            ax2.plot(self.his.sample_radius_his, label="Sample radius")
        ax3.plot(self.his.grad_norm_his)
        ax4.plot(self.his.rho_his)

        ax1.set_xlabel("Number of iterations")
        ax1.set_ylabel("Function value")
        ax2.set_xlabel("Number of iterations")
        ax2.set_ylabel("Tr radius")
        ax3.set_xlabel("Number of iterations")
        ax3.set_ylabel("Gradient norm")
        ax4.set_xlabel("Number of iterations")
        ax4.set_ylabel("Rho")

        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax3.set_yscale("log")
        ax4.set_yscale("log")

        ax2.legend()

        fig.suptitle(f"DRSOM training information")
        fig.subplots_adjust(wspace=0.4, hspace=0.4)

        folder = f"Subspace/fig"
        if not os.path.exists(folder):
            os.makedirs(folder)

        fig.savefig(f"{folder}/temp.png", dpi=1000)
        plt.close()
        return 0

    # NOTE: this function should be used after one step
    def draw_landscape(self, closure, id=0, resolution=50, padding=0.1):
        if self.sub_dim != 2:
            print(f"Subspace dimension != 2: {self.sub_dim}")
            return 0

        params = self._clone_param()
        self._set_param(self.params_copy)

        # get two basis
        d1 = self.basis[0]
        d2 = self.basis[1]

        # get three points
        star_x = [
            torch.dot(self.d_star, d1).detach().cpu().item(),
            torch.dot(self.d_star, d2).detach().cpu().item(),
        ]
        init_x = [0, 0]
        if self.rho > neta:
            final_x = star_x
        else:
            final_x = init_x

        # get dynamic range
        if self.construct_model_opt == 0:
            r = self.r
        else:
            r = 0

        def get_dynamic_range(opt, axes):
            if opt == 0:
                fig_r = 0.5 * torch.norm(self.d_star).cpu().item()
                all_x1_vals = [0, star_x[0], fig_r, -fig_r]
                all_x2_vals = [0, star_x[1], fig_r, -fig_r]
                min_x1, max_x1 = min(all_x1_vals), max(all_x1_vals)
                min_x2, max_x2 = min(all_x2_vals), max(all_x2_vals)
                range_x1 = (
                    min_x1 - padding * abs(max_x1 - min_x1),
                    max_x1 + padding * abs(max_x1 - min_x1),
                )
                range_x2 = (
                    min_x2 - padding * abs(max_x2 - min_x2),
                    max_x2 + padding * abs(max_x2 - min_x2),
                )
                x1_vals = np.linspace(range_x1[0], range_x1[1], resolution)
                x2_vals = np.linspace(range_x2[0], range_x2[1], resolution)
            else:
                all_x1_vals = [0, r, -r]
                all_x2_vals = [0, r, -r]
                min_x1, max_x1 = min(all_x1_vals), max(all_x1_vals)
                min_x2, max_x2 = min(all_x2_vals), max(all_x2_vals)
                range_x1 = (
                    min_x1 - padding * abs(max_x1 - min_x1),
                    max_x1 + padding * abs(max_x1 - min_x1),
                )
                range_x2 = (
                    min_x2 - padding * abs(max_x2 - min_x2),
                    max_x2 + padding * abs(max_x2 - min_x2),
                )
                x1_vals = np.linspace(range_x1[0], range_x1[1], resolution)
                x2_vals = np.linspace(range_x2[0], range_x2[1], resolution)
            for ax in axes:
                ax.set_xlim(range_x1[0], range_x1[1])
                ax.set_ylim(range_x2[0], range_x2[1])
            return x1_vals, x2_vals

        # get landscape values
        def get_landscape_values(x1_vals, x2_vals):
            z_exact_vals = np.zeros((resolution, resolution))
            z_model_vals = np.zeros((resolution, resolution))

            v_ls = [d1, d2]
            for i, x1 in enumerate(x1_vals):
                for j, x2 in enumerate(x2_vals):
                    coeff = np.array([x1, x2])
                    coeff_tensor = torch.tensor(
                        coeff, device=self.device, dtype=torch.float32
                    )
                    d_new = self.add_together(v_ls, coeff_tensor)
                    z_exact_vals[j, i] = (
                        self.get_value(d_new, closure).detach().cpu().item()
                    )
                    z_model_vals[j, i] = self.get_model_val(coeff.reshape(-1, 1))

            vmin = min(z_exact_vals.min(), z_model_vals.min())
            vmax = max(z_exact_vals.max(), z_model_vals.max())

            return z_exact_vals, z_model_vals, vmin, vmax

        # draw landscape and trajectories
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        (ax1, ax2), (ax3, ax4) = axes

        # part 1
        x1_vals, x2_vals = get_dynamic_range(opt=0, axes=(ax1, ax2))
        z_exact_vals, z_model_vals, vmin, vmax = get_landscape_values(x1_vals, x2_vals)

        contour1 = ax1.contourf(
            x1_vals,
            x2_vals,
            z_exact_vals,
            levels=100,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
        square.set_clip_box(ax1.bbox)
        ax1.add_patch(square)
        ax1.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )
        ax1.scatter(
            final_x[0],
            final_x[1],
            color="tab:orange",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter}}}$",
        )
        ax1.scatter(
            star_x[0],
            star_x[1],
            color="tab:red",
            marker="*",
            s=15,
            label="Model Solution",
        )

        _ = ax2.contourf(
            x1_vals,
            x2_vals,
            z_model_vals,
            levels=100,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
        square.set_clip_box(ax2.bbox)
        ax2.add_patch(square)
        ax2.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )
        ax2.scatter(
            final_x[0],
            final_x[1],
            color="tab:orange",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter}}}$",
        )
        ax2.scatter(
            star_x[0],
            star_x[1],
            color="tab:red",
            marker="*",
            s=15,
            label="Model Solution",
        )

        # part 2
        x1_vals, x2_vals = get_dynamic_range(opt=1, axes=(ax3, ax4))
        z_exact_vals, z_model_vals, vmin, vmax = get_landscape_values(x1_vals, x2_vals)

        contour3 = ax3.contourf(
            x1_vals,
            x2_vals,
            z_exact_vals,
            levels=20,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
        ax3.add_patch(square)
        ax3.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )

        _ = ax4.contourf(
            x1_vals,
            x2_vals,
            z_model_vals,
            levels=20,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        square = patches.Rectangle((-r, -r), 2 * r, 2 * r, fill=False, color="red")
        ax4.add_patch(square)
        ax4.scatter(
            0,
            0,
            color="tab:green",
            marker="o",
            s=30,
            label=rf"$x_{{{self.his.niter-1}}}$",
        )

        ax1.legend()
        ax1.set_xlabel("First direction")
        ax1.set_ylabel("Second direction")
        ax1.set_title("Exact value")

        ax2.legend()
        ax2.set_xlabel("First direction")
        ax2.set_ylabel("Second direction")
        ax2.set_title("Model value")

        ax3.legend()
        ax3.set_xlabel("First direction")
        ax3.set_ylabel("Second direction")
        ax3.set_title("Exact value")

        ax4.legend()
        ax4.set_xlabel("First direction")
        ax4.set_ylabel("Second direction")
        ax4.set_title("Model value")

        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        cbar1 = fig.colorbar(
            contour1, ax=(ax1, ax2), orientation="vertical", label="Function value"
        )
        cbar2 = fig.colorbar(
            contour3, ax=(ax3, ax4), orientation="vertical", label="Function value"
        )
        fig.suptitle(f"DRSOM Landscape and Optimization Trajectories")

        for ax in axes.flat:
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
        cbar1.ax.ticklabel_format(style="sci", scilimits=(-2, 2))
        cbar2.ax.ticklabel_format(style="sci", scilimits=(-2, 2))

        folder = f"Subspace/fig/landscape"
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(f"{folder}/{id}.png", dpi=500)
        plt.close()

        self._set_param(params)
        self.loss = closure(backward=False)
        self.y_current = self.loss.detach()
