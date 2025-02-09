"""
Utils for monkey patching across spawned processes.
"""

from multiprocessing.process import BaseProcess


patches_applied = {}


def multi_process_patcher(patch_func):
    """
    Run patch_func on this process,
    dnd on all spawned processes from this point on.
    """
    if patch_func in patches_applied:
        return
    patches_applied[patch_func] = None

    patch_func()
    start = BaseProcess.start

    def patched_start(self):
        self._target = TargetWrapper(self._target, patch_func)
        return start(self)

    BaseProcess.start = patched_start


class TargetWrapper:
    """
    Wrapper of a process target function which runs a patch function.
    """

    def __init__(self, target, patch_func):
        """
        init a wrapper instance.
        """
        self.target = target
        self.patch_func = patch_func

    def __call__(self, *args, **kwargs):
        """
        invoke the wrapped target.
        """
        multi_process_patcher(self.patch_func)
        return self.target(*args, **kwargs)
