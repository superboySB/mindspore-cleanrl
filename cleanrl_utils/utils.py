import mindspore as ms
from mindspore import nn
import collections


def sync_weight(model: nn.Cell, model_old: nn.Cell):
    """Synchronize the weight for the target network."""
    params_dict = model.parameters_dict()
    old_params_dict = model_old.parameters_dict()
    modified_dict = collections.OrderedDict(
        [(k_o, v) for (k, v), (k_o, v_o) in zip(params_dict.items(), old_params_dict.items())])
    not_loaded = ms.load_param_into_net(model_old, modified_dict)  # 可返回网络中没有被加载的参数。
    return not_loaded
