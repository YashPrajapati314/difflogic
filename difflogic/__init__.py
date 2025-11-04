try:
    from .difflogic import LogicLayer, GroupSum
except (ModuleNotFoundError, ImportError):
    LogicLayer, GroupSum = None, None
from .packbitstensor import PackBitsTensor
from .compiled_model import CompiledLogicNet

