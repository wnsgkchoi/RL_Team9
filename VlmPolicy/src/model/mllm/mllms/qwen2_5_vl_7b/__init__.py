from model.mllm.mllms.qwen2_5_vl_7b._base import Qwen2_5_VL_7B, Qwen2_5_VL_7B_Factory
from model.mllm.mllms.qwen2_5_vl_7b.local import Qwen2_5_VL_Local
from model.mllm.mllms.qwen2_5_vl_7b.client import Qwen2_5_VL_Client
from model.mllm.utils.constants import LoadMode

# Register implementations with the factory
Qwen2_5_VL_7B_Factory.register(LoadMode.LOCAL, Qwen2_5_VL_Local)
Qwen2_5_VL_7B_Factory.register(LoadMode.REMOTE, Qwen2_5_VL_Client)

__all__ = [
    "Qwen2_5_VL_7B",
    "Qwen2_5_VL_7B_Factory",
    "Qwen2_5_VL_Local",
    "Qwen2_5_VL_Client",
]
