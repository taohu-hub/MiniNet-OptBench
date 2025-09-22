import inspect
from torch.optim import Adam, AdamW, SGD

# Robust Muon imports to handle both layouts
MUON_IMPORT_OK = False
Muon = SingleDeviceMuon = MuonWithAuxAdam = SingleDeviceMuonWithAuxAdam = None
try:
    from opts.muon import (
        Muon, SingleDeviceMuon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
    )
    MUON_IMPORT_OK = True
except Exception:
    try:
        from muon import (
            Muon, SingleDeviceMuon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
        )
        MUON_IMPORT_OK = True
    except Exception:
        MUON_IMPORT_OK = False

# SciPy is optional; we'll gracefully fall back to NumPy SVD for condition numbers
try:
    from scipy import linalg as sp_linalg
except Exception:
    sp_linalg = None

# -----------------------------------------------------------------------------
# Optimizer initialization (upper-file style, extended to cover lower-file use)
# -----------------------------------------------------------------------------
def initialize_optimizer(model, optimizer_class_obj, param_groups_config, device_type, ddp=False):
    """Initialize optimizer instance from a param-group *config* (upper-file style).

    - For Adam/AdamW/SGD: builds groups from the config and creates the optimizer.
    - For Muon: supports distributed/single-device classes.
    - For MuonWithAuxAdam: pass group dicts through (must carry 'use_muon' flags).

    param_groups_config: list of dicts. Each dict may contain:
        - group_type: 'decay'|'nodecay'|'hidden'|'other'|'all' (required to map params)
        - Any hyperparams for the target optimizer (e.g., lr, betas, eps, weight_decay,
          momentum, use_muon, etc.).
    """
    # 1) collect trainable parameters
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # 2) build actual param groups by 'group_type'
    built_groups = []
    for gc in param_groups_config:
        gtype = gc.get('group_type', None)
        g = {k: v for k, v in gc.items() if k != 'group_type'}
        if gtype == 'decay':
            params = [p for n, p in param_dict.items() if p.dim() >= 2]
        elif gtype == 'nodecay':
            params = [p for n, p in param_dict.items() if p.dim() < 2]
        elif gtype == 'hidden':
            params = [p for n, p in param_dict.items() if p.ndim >= 2 and "embed" not in n and "lm_head" not in n]
        elif gtype == 'other':
            params = [p for n, p in param_dict.items() if not (p.ndim >= 2 and "embed" not in n and "lm_head" not in n)]
        elif gtype == 'all':
            params = list(param_dict.values())
        else:
            raise ValueError(f"Unknown group_type: {gtype}")
        g['params'] = params
        built_groups.append(g)

    cls_name = optimizer_class_obj.__name__
    if cls_name in ['Adam', 'AdamW']:
        opt = optimizer_class_obj(built_groups)  # pass group dicts directly
        fused_available = 'fused' in inspect.signature(optimizer_class_obj).parameters
        use_fused = fused_available and device_type == 'cuda'
        print(f"Using {cls_name} optimizer, fused={use_fused}")
        return opt

    elif cls_name in ['SGD']:
        # sanitize groups to valid SGD keys
        sanitized_groups = []
        for g in built_groups:
            ng = {k: v for k, v in g.items()
                  if k in ['params', 'learning_rate', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'foreach']}
            if 'momentum' not in ng:
                ng['momentum'] = 0.0
            sanitized_groups.append(ng)
        opt = optimizer_class_obj(sanitized_groups)
        print("Using SGD optimizer")
        return opt

    elif cls_name in ['Muon', 'SingleDeviceMuon']:
        # Expect one group that holds the global hyperparams
        g0 = built_groups[0] if built_groups else {'params': list(param_dict.values())}
        opt = optimizer_class_obj(
            g0['params'],
            lr=g0.get('learning_rate', 1e-3),
            weight_decay=g0.get('weight_decay', 0.0),
            momentum=g0.get('momentum', 0.9),
        )
        print(f"Using {'distributed' if ddp else 'single-device'} {cls_name} optimizer")
        return opt

    elif cls_name in ['MuonWithAuxAdam', 'SingleDeviceMuonWithAuxAdam']:
        # Pass through groups (must contain 'use_muon' to distinguish)
        print(f"Using {'distributed' if ddp else 'single-device'} {cls_name} optimizer")
        return optimizer_class_obj(built_groups)

    else:
        raise ValueError(f"Unsupported optimizer class: {cls_name}")

# -----------------------------------------------------------------------------
# Optimizer creation (preserve both APIs)
# -----------------------------------------------------------------------------
# Map upper-file optimizer_name strings to classes (Muon resolved later)
_opt_map = {
    'ADAM': Adam,
    'ADAMW': AdamW,
    'SGD': SGD,
    'MUON': 'MUON',  # Placeholder: resolves to Muon (ddp=True) or SingleDeviceMuon (ddp=False)
    'MUON_WITH_AUX_ADAM': 'MUON_WITH_AUX_ADAM',  # Placeholder: resolves to MuonWithAuxAdam (ddp=True) or SingleDeviceMuonWithAuxAdam (ddp=False)
    'SINGLE_DEVICE_MUON': 'SINGLE_DEVICE_MUON',  # Explicit single-device only (overrides ddp check)
    'SINGLE_DEVICE_MUON_WITH_AUX_ADAM': 'SINGLE_DEVICE_MUON_WITH_AUX_ADAM',  # Explicit single-device only (overrides ddp check)
}

def _default_adamw_groups(weight_decay, learning_rate, beta1, beta2):
    return [
        {'group_type': 'decay',   'weight_decay': weight_decay, 'learning_rate': learning_rate, 'betas': (beta1, beta2), 'eps': 1e-8},
        {'group_type': 'nodecay', 'weight_decay': 0.0,          'learning_rate': learning_rate, 'betas': (beta1, beta2), 'eps': 1e-8},
    ]

def _pick_optimizer_class_and_groups(weight_decay, learning_rate, beta1, beta2, optimizer_class, param_groups, optimizer_name, ddp, momentum, optimizer_type, use_muon_for_hidden_only):
    """Return (optimizer_class_obj, param_groups_config, chosen_name)"""
    # precedence: explicit class > name > type
    if optimizer_class is not None:
        return optimizer_class, (param_groups or _default_adamw_groups(weight_decay, learning_rate, beta1, beta2)), optimizer_class.__name__

    # name path (upper-file style)
    if optimizer_name:
        uname = optimizer_name.upper()
        klass = _opt_map.get(uname, None)
        if klass is None:
            raise ValueError(f"Unknown optimizer_name: {optimizer_name}")
        if isinstance(klass, str):  # Muon family
            if not MUON_IMPORT_OK:
                raise ImportError("Muon optimizer requested but Muon classes are not importable.")
            if 'WITH_AUX_ADAM' in klass:
                oc = MuonWithAuxAdam if ddp else SingleDeviceMuonWithAuxAdam
                pg = param_groups if param_groups is not None else [
                    {'group_type': 'other',  'use_muon': False, 'learning_rate': learning_rate, 'betas': (beta1, beta2), 'eps': 1e-8, 'weight_decay': weight_decay},
                    {'group_type': 'hidden', 'use_muon': True,  'learning_rate': learning_rate,      'momentum': momentum, 'eps': 1e-8, 'weight_decay': weight_decay},
                ]
                return oc, pg, uname
            else:
                oc = Muon if ddp else SingleDeviceMuon
                pg = param_groups if param_groups is not None else [
                    {'group_type': 'all', 'learning_rate': learning_rate, 'momentum': momentum, 'eps': 1e-8, 'weight_decay': weight_decay},
                ]
                return oc, pg, uname
        else:
            pg = param_groups if param_groups is not None else _default_adamw_groups(weight_decay, learning_rate, beta1, beta2)
            return klass, pg, uname

    if optimizer_type:
        t = optimizer_type.lower()
        if t == 'adamw' or t == 'adam':
            oc = AdamW if t == 'adamw' else Adam
            pg = param_groups if param_groups is not None else _default_adamw_groups(weight_decay, learning_rate, beta1, beta2)
            return oc, pg, oc.__name__
        elif t == 'muon':
            if not MUON_IMPORT_OK:
                raise ImportError("optimizer_type='muon' requested but Muon classes are not importable.")
            if use_muon_for_hidden_only:
                oc = MuonWithAuxAdam if ddp else SingleDeviceMuonWithAuxAdam
                pg = param_groups if param_groups is not None else [
                    {'group_type': 'other',  'use_muon': False, 'learning_rate': learning_rate, 'betas': (beta1, beta2), 'eps': 1e-8, 'weight_decay': weight_decay},
                    {'group_type': 'hidden', 'use_muon': True,  'learning_rate': learning_rate,      'momentum': momentum, 'eps': 1e-8, 'weight_decay': weight_decay},
                ]
                return oc, pg, oc.__name__
            else:
                oc = Muon if ddp else SingleDeviceMuon
                pg = param_groups if param_groups is not None else [
                    {'group_type': 'all', 'learning_rate': learning_rate, 'momentum': momentum, 'eps': 1e-8, 'weight_decay': weight_decay},
                ]
                return oc, pg, oc.__name__
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    # default: AdamW upper-file style
    return AdamW, (param_groups or _default_adamw_groups(weight_decay, learning_rate, beta1, beta2)), 'ADAMW'
