"""Opt-in CPU handoff for dense, single-device Torch HF joint training.

``lambda: release_training_models(bundles, optimizer, teacher)`` is the factory
accepted by serving_dispatch. The caller must stop feature production and reach
an optimizer boundary with no live forward/backward graph or retained tensor
views before entering. This context does not advance/clear queues or cursors.
The external verifier must terminate before context exit, freeing its CUDA memory.

Supports the ordinary OfflineTransformersCapture Llama/Qwen2/Qwen3 decoder path
(including MiniCPM exports using those HF architectures), BF16Optimizer/AdamW,
and dense DSpark/DFlash/DFlash2 bundles. Rejects packed/quantized tensors, tensor
subclasses, distinct objects sharing storage, views, distributed/sharded/offload
hooks and other accelerators before mutation. Same-object tied weights are kept.

Existing suspend_training_state owns optimizer/gradient/mode/buffer/RNG restore;
this inner layer moves model tensor DATA, never Parameter/module objects. CPU is
supported for local restoration tests but is not CUDA memory-release evidence.
CUDA restoration allocation/driver failure raises: caller must not commit a
verified decision and may need checkpoint recovery. No server/API is launched.
"""
from contextlib import contextmanager
import threading

import torch

from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.optimizer import BF16Optimizer

_ACTIVE = set()
_LOCK = threading.Lock()


def _leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)


def _validate(bundles, optimizer, teacher):
    if type(teacher) is not OfflineTransformersCapture:
        raise ValueError('ordinary Torch HF capture required; Prism/packed teachers unsupported')
    model = teacher._model
    if (not isinstance(model, torch.nn.Module) or model.config.model_type not in ('llama', 'qwen2', 'qwen3')
            or not type(model).__module__.startswith('transformers.models.')):
        raise ValueError('unsupported HF teacher architecture')
    if type(optimizer) is not BF16Optimizer or type(optimizer.optimizer) is not torch.optim.AdamW:
        raise ValueError('dense BF16Optimizer/AdamW required')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('multi-rank serving handoff unsupported')
    roots = [model]
    if not bundles:
        raise ValueError('nonempty dense joint bundles required')
    for bundle in bundles:
        for name in ('model', 'draft_model'):
            root = getattr(bundle, name, None)
            if not isinstance(root, torch.nn.Module):
                raise ValueError('dense joint modules required')
            roots.append(root)
        if getattr(bundle, 'target_head', None) is not None:
            if not isinstance(bundle.target_head, torch.nn.Module):
                raise ValueError('dense target head required')
            roots.append(bundle.target_head)
    modules = {id(m): m for root in roots for m in root.modules()}
    tensors = {id(t): t for m in modules.values()
               for t in [*m._parameters.values(), *m._buffers.values()] if t is not None}
    for module in modules.values():
        kind = (type(module).__module__ + '.' + type(module).__name__).lower()
        if (any(word in kind for word in ('prism', 'packed', 'quantiz', 'fullysharded', 'distributeddataparallel'))
                or hasattr(module, '_hf_hook') or hasattr(module, '_orig_mod')
                or getattr(module, 'is_loaded_in_4bit', False) or getattr(module, 'is_loaded_in_8bit', False)):
            raise ValueError('packed/quantized/sharded/hooked models unsupported')
        # Ordinary rotary aliases can reference a registered tensor by identity.
        # An extra CUDA cache/weight must not silently remain resident.
        for value in vars(module).values():
            for tensor in _leaves(value):
                if tensor.device.type != 'cpu' and id(tensor) not in tensors:
                    raise ValueError('unregistered accelerator tensor/storage unsupported')
    all_tensors = dict(tensors)
    for tensor in [*optimizer.fp32_params, *optimizer.model_params,
                   *_leaves(optimizer.optimizer.state), *_leaves(optimizer.last_grad_norm)]:
        all_tensors[id(tensor)] = tensor
    for tensor in list(all_tensors.values()):
        if tensor.grad is not None:
            all_tensors[id(tensor.grad)] = tensor.grad
    storage, devices = {}, set()
    for tensor in all_tensors.values():
        if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.device.type not in ('cpu', 'cuda')
                or not tensor.is_leaf or tensor._base is not None or not tensor.is_contiguous()
                or tensor.storage_offset() != 0):
            raise ValueError('dense owning CPU/CUDA tensors required; views/subclasses unsupported')
        if tensor.numel():
            if tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
                raise ValueError('partial/oversized tensor storage unsupported')
            key = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if key in storage and storage[key] != id(tensor):
                raise ValueError('distinct tensor objects sharing storage unsupported')
            storage[key] = id(tensor)
        if tensor.device.type == 'cuda':
            devices.add(tensor.device)
    if len(devices) > 1:
        raise ValueError('multi-device serving handoff unsupported')
    return modules, list(tensors.values()), devices


def _move_data(tensor, device):
    tensor.data = tensor.detach().to(device=device)


@contextmanager
def release_training_models(bundles, optimizer, teacher):
    """Release registered model CUDA storage, then restore even on verifier failure."""
    from draftfit.training.joint_verification import suspend_training_state

    bundles = tuple(bundles)
    modules, tensors, devices = _validate(bundles, optimizer, teacher)
    identities = {*modules, id(optimizer)}
    with _LOCK:
        if _ACTIVE.intersection(identities):
            raise ValueError('overlapping serving memory handoff')
        _ACTIVE.update(identities)
    try:
        for device in devices:
            torch.cuda.synchronize(device)
        before = {str(d): torch.cuda.memory_allocated(d) for d in devices}
        model_bytes = sum(t.numel() * t.element_size() for t in tensors if t.device.type == 'cuda')
        with suspend_training_state(bundles, optimizer, teacher, cpu_offload=True):
            moved = []
            try:
                with torch.no_grad():
                    for tensor in tensors:
                        if tensor.device.type == 'cuda':
                            device = tensor.device
                            _move_data(tensor, 'cpu')
                            moved.append((tensor, device))
                for device in devices:
                    torch.cuda.synchronize(device)
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                after = {str(d): torch.cuda.memory_allocated(d) for d in devices}
                if any(t.device.type != 'cpu' for t in tensors):
                    raise RuntimeError('model tensor still resident on accelerator')
                if devices and not all(after[str(d)] < before[str(d)] for d in devices):
                    raise RuntimeError('CUDA allocator did not release training memory')
                yield dict(mode='released_training_memory', target_resident=False,
                           model_tensor_count=len(tensors), model_cuda_bytes=model_bytes,
                           cuda_allocated_before=before, cuda_allocated_after=after,
                           cuda_release_observed=bool(devices))
            finally:
                errors = []
                with torch.no_grad():
                    for tensor, device in reversed(moved):
                        try:
                            _move_data(tensor, device)
                        except Exception as exc:
                            errors.append(exc)
                if errors:
                    raise RuntimeError('model memory restore failed; checkpoint recovery required') from errors[0]
        for device in devices:
            torch.cuda.synchronize(device)
    finally:
        with _LOCK:
            _ACTIVE.difference_update(identities)


__all__ = ['release_training_models']
