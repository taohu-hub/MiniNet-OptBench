# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch implementation of Shampoo."""

from __future__ import print_function

import enum
import itertools

from dataclasses import dataclass
from . import matrix_functions
import numpy as np
import torch
import torch.optim as optim


# Grafting is a technique to fix the layerwise scale of Shampoo optimizer.
# https://arxiv.org/pdf/2002.11803.pdf studies this in detail. This
# allows us to plugin the Shampoo optimizer into settings where SGD/AdaGrad
# is already well tuned. Grafting onto Shampoo means take the Shampoo direction,
# but use the step magnitude from the grafted optimizer such as Adagrad or SGD.
class LayerwiseGrafting(enum.IntEnum):
  NONE = 0
  SGD = 1
  ADAGRAD = 2


@dataclass
class ShampooHyperParams:
  """Shampoo hyper parameters."""
  beta2: float = 1.0
  diagonal_eps: float = 1e-6
  matrix_eps: float = 1e-12
  Q_eps: float = 1e-6
  weight_decay: float = 0.0
  inverse_exponent_override: int = 0  # fixed exponent for preconditioner, if >0
  start_preconditioning_step: int = 1
  # Performance tuning params for controlling memory and compute requirements.
  # How often to compute preconditioner.
  preconditioning_compute_steps: int = 1
  # How often to compute statistics.
  statistics_compute_steps: int = 1
  # Block size for large layers (if > 0).
  # Block size = 1 ==> Adagrad (Don't do this, extremely inefficient!)
  # Block size should be as large as feasible under memory/time constraints.
  block_size: int = 128
  # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
  # 12 x [1024, 512] L and R statistics. Disabled by default which results in
  # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
  best_effort_shape_interpretation: bool = True
  # Type of grafting (SGD or AdaGrad).
  # https://arxiv.org/pdf/2002.11803.pdf
  graft_type: int = LayerwiseGrafting.SGD
  # Nesterov momentum
  nesterov: bool = True


class Graft:
  """Base class to perform grafting onto Shampoo. This class does no grafting.
  """

  def __init__(self, hps, unused_var):
    self.hps = hps

  def add_statistics(self, grad):
    pass

  def precondition_gradient(self, grad):
    # print("doing naive precondition gradient")
    # input("Press Enter to Continue")
    return grad

  def update_momentum(self, update, unused_beta1):
    return update


class SGDGraft(Graft):
  """Graft using SGD+momentum.

  momentum maintains an exponentially weighted moving average of gradients.
  """

  def __init__(self, hps, var):
    super(SGDGraft, self).__init__(hps, var)
    self.momentum = torch.zeros_like(var.data, device=var.get_device())

  def update_momentum(self, update, beta1):
    self.momentum.mul_(beta1).add_(update)
    return self.momentum


class AdagradGraft(SGDGraft):
  """Graft using Adagrad.

  Essentially an implementation of Adagrad with momentum.
  """

  def __init__(self, hps, var):
    super(AdagradGraft, self).__init__(hps, var)
    self.statistics = torch.zeros_like(var.data, device=var.get_device())

  def add_statistics(self, grad):
    self.statistics.add_(grad * grad)

  def precondition_gradient(self, grad):
    # print("doing Adagrad precondition_gradient")
    # print(self.statistics)
    # input("Press Enter to Continue")
    return grad / (torch.sqrt(self.statistics) + self.hps.diagonal_eps)


class BlockPartitioner:
  """Partitions a tensor into smaller tensors for preconditioning.

    For example, if a variable has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 variables of size
    (1024, 512) each.
  """

  def __init__(self, var, hps):
    self._shape = var.shape
    self._splits = []
    self._split_sizes = []
    split_sizes = []
    # We split var into smaller blocks. Here we store the metadata to make
    # that split.
    for i, d in enumerate(var.shape):
      if hps.block_size > 0 and d > hps.block_size:
        # d-1, otherwise split appends a 0-size array.
        nsplit = (d-1) // hps.block_size
        indices = (np.arange(nsplit, dtype=np.int32) + 1) * hps.block_size
        sizes = np.ones(nsplit + 1, dtype=np.int32) * hps.block_size
        sizes[-1] = d - indices[-1]
        self._splits.append((i, indices))
        self._split_sizes.append((i, sizes))
        split_sizes.append(sizes)
      else:
        split_sizes.append(np.array([d], dtype=np.int32))
    self._num_splits = len(split_sizes)
    self._preconditioner_shapes = []
    for t in itertools.product(*split_sizes):
      self._preconditioner_shapes.extend([[d, d] for d in t])

  def shapes_for_preconditioners(self):
    return self._preconditioner_shapes

  def num_splits(self):
    return self._num_splits

  def partition(self, tensor): # This partition is trivial, it simply put the tensor to a one-element list.
    """Partition tensor into blocks."""

    assert tensor.shape == self._shape
    tensors = [tensor]
    # print("We are now doing partition for tensor")
    # print(tensor.shape)
    # print(tensor)
    # print(self._split_sizes)
    for (i, sizes) in self._split_sizes:
      tensors_local = []
      for t in tensors:
        tensors_local.extend(
            torch.split(t, tuple(sizes), dim=i))
      tensors = tensors_local
    # print("the partitioned tensor is")
    # print(len(tensors))
    # print(tensors)
    return tensors

  def merge_partitions(self, partitions):
    """Merge partitions back to original shape."""

    for (i, indices) in reversed(self._splits):
      n = len(indices) + 1
      partial_merged_tensors = []
      ind = 0
      while ind < len(partitions):
        partial_merged_tensors.append(
            torch.cat(partitions[ind:ind + n], axis=i))
        ind += n
      partitions = partial_merged_tensors
    assert len(partitions) == 1
    return partitions[0]


def _merge_small_dims(shape_to_merge, max_dim):#Try to achieve a trade-off between the dimensionality of the tensor and the size of the tensor.
  """Merge small dimensions.

  If there are some small dimensions, we collapse them:
  e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
       [1, 2, 768, 1, 2048] --> [2, 768, 2048]

  Args:
    shape_to_merge: Shape to merge small dimensions.
    max_dim: Maximal dimension of output shape used in merging.

  Returns:
    Merged shape.
  """
  resulting_shape = []
  product = 1
  for d in shape_to_merge:
    if product * d <= max_dim:
      product *= d
    else:
      if product > 1:
        resulting_shape.append(product)
      product = d
  if product > 1:
    resulting_shape.append(product)
  return resulting_shape


class Preconditioner:
  """Compute statistics/shape from gradients for preconditioning."""

  def __init__(self, var, hps):
    self._hps = hps
    self._original_shape = var.shape
    self._transformed_shape = var.shape
    if hps.best_effort_shape_interpretation:
      # print("We are now using best effort shape interpretation.")
      self._transformed_shape = _merge_small_dims(
          self._original_shape, hps.block_size) # merge all SMAL dimensions
    
    # print(f"var.shape = {var.shape}")
    # print(f"self._transformed_shape{self._transformed_shape}")
    # print(f"self._original_shape{self._original_shape}")

    reshaped_var = torch.reshape(var, self._transformed_shape)
    self._partitioner = BlockPartitioner(reshaped_var, hps) # partition all BIG dimensions
    shapes = self._partitioner.shapes_for_preconditioners()
    rank = len(self._transformed_shape)
    # print(f"rank={rank} , self._transformed_shape{self._transformed_shape}")
    # input("Press Enter to Continue")
    device = var.get_device()
    if rank <= 1:
      self.statistics = []
      self.preconditioners = []
    else:
      eps = self._hps.matrix_eps
      self.statistics = [eps * torch.eye(s[0], device=device) for s in shapes]
      self.preconditioners = [torch.eye(s[0], device=device) for s in shapes] # The default preconditioner is no preconditioner

  def add_statistics(self, grad):#L_t = w1 * L_{t - 1} + w2 * grad \otimes grad 
    """Compute statistics from gradients and add to the correct state entries.

    Args:
      grad: Gradient to compute statistics from.
    """
    if not self.statistics: return
    # print("Now we are adding statistics")
    reshaped_grad = torch.reshape(grad, self._transformed_shape)
    # print(self._transformed_shape)
    partitioned_grads = self._partitioner.partition(reshaped_grad)
    w1 = self._hps.beta2
    w2 = 1.0 if w1 == 1.0 else (1.0 - w1)
    # print(f"w1 = {w1} , w2 = {w2}")
    rank = len(self._transformed_shape)
    for j, grad in enumerate(partitioned_grads):
      # if j > 0:
      #   print(partitioned_grads[0].shape , partitioned_grads[1].shape)
        # input("Press Enter to Continue")
      #assert j == 0, "My hypothesis that grad is always partitioned into one part is wrong!"
      #The partition procedure is to ensure that the length would not exceed 128.
      for i in range(rank):
        axes = list(range(i)) + list(range(i + 1, rank))
        stat = torch.tensordot(grad, grad, [axes, axes])
        self.statistics[j*rank + i].mul_(w1).add_(stat, alpha=w2)
        # print(f"The shape is {self.statistics[j*rank + i].shape}")

  def exponent_for_preconditioner(self):
    """Returns exponent to use for inverse-pth root M^{-1/p}."""
    if self._hps.inverse_exponent_override > 0:
      return self._hps.inverse_exponent_override
    return 2 * len(self._transformed_shape)

  def compute_preconditioners(self):
    """Compute L^{-1/exp} for each stats matrix L."""
    exp = self.exponent_for_preconditioner()
    eps = self._hps.matrix_eps
    for i, stat in enumerate(self.statistics):
      self.preconditioners[i] = matrix_functions.ComputePower(
          stat, exp, ridge_epsilon=eps)

  def preconditioned_grad(self, grad):
    """Precondition the gradient.

    Args:
      grad: A gradient tensor to precondition.

    Returns:
      A preconditioned gradient.
    """
    if not self.preconditioners: return grad
    # print("Now we are doing preconditioned_grad")
    # print(f"The shape will be tranform   
    # assert grad.shape[0] == grad.shape[1], "nonsquare comming in!"

    reshaped_grad = torch.reshape(grad, self._transformed_shape)
    # print(f"reshaped_grad.shape = {reshaped_grad.shape}")
    partitioned_grads = self._partitioner.partition(reshaped_grad)#I think this line has trival use here at least for shampoo
    preconditioned_partitioned_grads = []
    num_splits = self._partitioner.num_splits()
    for i, grad in enumerate(partitioned_grads):
      preconditioners_for_grad = self.preconditioners[i * num_splits:(i + 1) *
                                                      num_splits]
      rank = len(grad.shape)
      precond_grad = grad
      for j in range(rank):
        preconditioner = preconditioners_for_grad[j]
        # print(f"preconditioner.shape{preconditioner.shape}")
        precond_grad = torch.tensordot(
            precond_grad, preconditioner, [[0], [0]]) # precond_grad = L_t^{-1/4} grad R_t^{-1/4}, the power is computed in ComputePower
      preconditioned_partitioned_grads.append(precond_grad)
      # print(f"precon_grad.shape = {precond_grad.shape}")
    merged_grad = self._partitioner.merge_partitions(
        preconditioned_partitioned_grads) 
    # print(f"self._original_shape = {self._original_shape}")
    return torch.reshape(merged_grad, self._original_shape)


STEP = 'step'
MOMENTUM = 'momentum'
PRECONDITIONER = 'preconditioner'
GRAFT = 'graft'
SHAMPOOGRAD = 'shampoograd'

class Shampoo_drsom(optim.Optimizer):
  """The Shampoo optimizer."""

  def __init__(self,
               params,
               lr=5e-5,
               momentum=0.9,
               hyperparams=ShampooHyperParams(),
               statistics_compute_steps: int = -1,
               preconditioning_compute_steps: int = -1
               ):
    # print("We are going to init shampoo_drsom")
    if statistics_compute_steps != -1:
      hyperparams.statistics_compute_steps = int(statistics_compute_steps)
    if preconditioning_compute_steps != -1:
      hyperparams.preconditioning_compute_steps = int(preconditioning_compute_steps)
    # print("updated compute steps")
    defaults = dict(lr=lr, momentum=momentum)
    self.hps = hyperparams
    super(Shampoo_drsom, self).__init__(params, defaults)
    # print("already initiated")

  def _gather_flat_grad(self, loss , retain_the_computation_graph = True):
        # print("We are now gathering flat grad")
        # print(f"The loss is useable{loss.grad_fn}")

        params = [p for g in self.param_groups for p in g['params']]

        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        views_grad = []
        for p, g in zip(params, grads):
          # print(p)
          # input("Press Enter to Continue")
          # input("Press Enter to Continue")
          # print(g)
          if g is None:
            view_grad = p.new_zeros(p.numel(), requires_grad=True)
            assert view_grad.require_grad == True, "view_grad has no gradient? Impossible!"
          else:
            view_grad = g.contiguous().view(-1)
          # print(f"view_grad.shape{view_grad.shape}")
          views_grad.append(view_grad)

        # print("So far so good! Every stuffs are collected!")

        return torch.cat(views_grad, 0)

  def _gather_shampoo_grad_and_momentum(self):
    views_shampoo_grad = []
    views_mom = []
    for group in self.param_groups:
        for p in group['params']:
          state = self.state[p]
          if state[SHAMPOOGRAD] is None:
            view_shampoo_grad = p.new(p.numel()).zero_()
            assert view_shampoo_grad.require_grad == False, "view_shampoo_grad has gradient? Impossible!"
          else:
            view_shampoo_grad = state[SHAMPOOGRAD].view(-1)
          # print(f"view_shampoo_grad.shape{view_shampoo_grad.shape}")
          views_shampoo_grad.append(view_shampoo_grad)


          # print(f"state[MOMENTUM]{state[MOMENTUM]}")
          if state[MOMENTUM] is None:
            view_mom = p.new(p.numel()).zero_()
            assert view_mom.require_grad == False, "view_mom has gradient? Impossible!"
          else:
            view_mom = state[MOMENTUM].view(-1)
          # print(f"view_mom.shape{view_mom.shape}")
          views_mom.append(view_mom)

        # print("So far so good! Every stuffs are collected!")

        return torch.cat(views_shampoo_grad, 0) , torch.cat(views_mom , 0)

  def _add_grad(self, step_size, update):
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
            offset += numel
            # print(f"{numel} of the elements are added.")
            state[STEP] = state[STEP] + 1

  def init_var_state(self, var, state): # var is taken to be p
    """Initialize the PyTorch state of for a single variable."""
    # print("initializeing var state")
    # input("Press Enter to continue")
    state[STEP] = 0
    state[MOMENTUM] = torch.zeros_like(var.data, device=var.get_device())
    state[SHAMPOOGRAD] = torch.zeros_like(var.data , device = var.get_device())

    state[MOMENTUM] = state[MOMENTUM].detach()
    state[SHAMPOOGRAD] = state[SHAMPOOGRAD].detach()

    assert state[MOMENTUM].requires_grad == False, "MOMENTUM will interfere with the differentiation process"
    assert state[SHAMPOOGRAD].requires_grad == False, "SHAMPOOGRAD will interfere with the differentiation process"

    state[PRECONDITIONER] = Preconditioner(var, self.hps)
    if self.hps.graft_type == LayerwiseGrafting.ADAGRAD:
      state[GRAFT] = AdagradGraft(self.hps, var)
    elif self.hps.graft_type == LayerwiseGrafting.SGD:
      state[GRAFT] = SGDGraft(self.hps, var)
    else:
      state[GRAFT] = Graft(self.hps, var)

  def step(self, closure=None):
    hps = self.hps
    # print(f"hps: {hps}")
    # input("Press Enter to Continue")
    # print("get_hps_parameters")

    if closure is None:
        loss = None
    else:
        closure = torch.enable_grad()(closure)
        loss = closure()

    cnt = 0

    for group in self.param_groups:
      cnt = cnt + 1
      lr = group['lr']
      # print("get_lr")
      # print("Now group params will be printed")
      # print(group['params'])
      for p in group['params']:
        # print(f"Now p would be printed, which has size{p.shape}")
        # print(p)
        # print(p)
        if p.grad is None: continue
        grad = p.grad.data
        # print(f"Now the grad will be printed")
        # print(grad)
        # input("Press Enter to continue")
        if grad.is_sparse:
          raise RuntimeError('Shampoo does not support sparse yet')
        state = self.state[p]
        # print(f"p.shape = {p.shape}")
        if not state:
          self.init_var_state(p, state)
        state[STEP] += 1
        # print("var_state is initialized")

        preconditioner = state[PRECONDITIONER]
        graft = state[GRAFT]

        # print("preconditioner and graft are initialized")

        # Gather statistics, compute preconditioners
        graft.add_statistics(grad)
        # print("Grad is added to the graft")
        if state[STEP] % hps.statistics_compute_steps == 0:
          preconditioner.add_statistics(grad)
        # print("Grad is added to the preconditioner")
        if state[STEP] % hps.preconditioning_compute_steps == 0:
          preconditioner.compute_preconditioners()
        # print("preconditioner is computed.")

        # print("statistics are gathered.")

        # Precondition gradients
        # print("now the grafted gradient will be printed")
        graft_grad = graft.precondition_gradient(grad)
        # print("Now the preconditioned gradient would be printed.")
        # print(graft_grad)
        shampoo_grad = grad
        # print("now the grad will be printed")
        # print(f"grad.shape = {grad.shape} , state[STEP] = {state[STEP]}")
        if state[STEP] >= self.hps.start_preconditioning_step:
          shampoo_grad = preconditioner.preconditioned_grad(grad)
        # print(f"grad.shape{grad.shape} , graft_grad.shape{graft_grad.shape} , shampoo_grad{shampoo_grad.shape}")

        # Grafting
        graft_norm = torch.norm(graft_grad)
        shampoo_norm = torch.norm(shampoo_grad)
        shampoo_grad.mul_(graft_norm / (shampoo_norm + 1e-16))
        # print("The gradient norms are collected.")

        # Weight decay
        if self.hps.weight_decay != 0.0:
          shampoo_grad.add_(p.data, alpha=self.hps.weight_decay)
          graft_grad.add_(p.data, alpha=self.hps.weight_decay)

        state[SHAMPOOGRAD] = shampoo_grad
        # print("Weights are decayed!")

    flat_grad = self._gather_flat_grad(loss)
    # print("flat_grads are gathered!")
    flat_shampoo_grad, flat_momentum = self._gather_shampoo_grad_and_momentum()
    # print("flat vectors are gathered!")

    assert flat_grad.requires_grad == True, "Differentiating grad would not yield Hessian!"
    assert flat_shampoo_grad.requires_grad == False, "shampoo grad would interfere the result of Hessian vector product!"
    assert flat_momentum.requires_grad == False, "momentum would interfere the result of Hessian vector product!"

    # print("flat_grad.shape is :")
    # print(flat_grad.shape)
    # print("flat_shampoo_grad.shape is :")
    # print(flat_shampoo_grad.shape)
    # print("flat_momentum.shape is:")
    # print(flat_momentum.shape)
    grad_times_shampoo_grad = torch.dot(flat_grad , flat_shampoo_grad)
    # print("OK, we can dot shampoo grad and grad")
    grad_times_mom = torch.dot(flat_grad , flat_momentum)
  # print("OK, we can dot grad and momentum")


    Hessian_shampoo_grad = self._gather_flat_grad(grad_times_shampoo_grad , retain_the_computation_graph=True)
    Hessian_mom = self._gather_flat_grad(grad_times_mom , retain_the_computation_graph=True)

    # print(f"Hessian_shampoo_grad.shape={Hessian_shampoo_grad.shape}")
    # print(f"Hessian_mom.shape={Hessian_mom.shape}")

    shampoo_grad_Hessian_shampoo_grad = torch.dot(flat_shampoo_grad , Hessian_shampoo_grad)
    # print("OK, we can shampoo H shampoo")
    mom_Hessian_mom = torch.dot(flat_momentum , Hessian_mom)
    # print("OK, we can mom_H_mom")
    shampoo_grad_Hessian_mom = torch.dot(flat_momentum , Hessian_shampoo_grad)
    # print("OK, we can shampoo H mom")

    subspace_hessian = torch.tensor([[shampoo_grad_Hessian_shampoo_grad , -shampoo_grad_Hessian_mom], [-shampoo_grad_Hessian_mom, mom_Hessian_mom]], device=flat_grad.device)
    c = torch.tensor([[- grad_times_shampoo_grad ], [grad_times_mom]], device=flat_grad.device)

    # print("subspace_hessian and c are calculated.")

    with torch.no_grad():
      cond_number = float(torch.linalg.cond(subspace_hessian))
      det_Q = float(torch.linalg.det(subspace_hessian))

    # print("cond_number and det_Q are calculated")
    # print(cond_number)
    # print(det_Q)

    if det_Q != 0 and cond_number < 1/hps.Q_eps:
      # print("The if is performed smoothly!")
      with torch.no_grad():
        alpha = torch.linalg.solve(subspace_hessian, -c)
      # print("alpha is calculated!")

      d = -alpha[0] * flat_shampoo_grad + alpha[1] * flat_momentum
    else:
      # print("Now the matrix is singular, we are now using gradient descent.")
      d = - flat_shampoo_grad

    # print("The direction would be cloned.")
    d = d.detach().clone()

    # print("The direction would be added.")
    self._add_grad(lr, d)

    # print("The direction is added.")

    assert cnt == 1, "There is more than one group!"

    # return loss
    # print(f"loss: {loss}")
    # input("Press Enter to Continue")
