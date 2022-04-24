from tkinter import HIDDEN
from numpy import require

import torch
import torch.nn as nn
import numpy as np
import FrEIA.modules as Fm
import FrEIA.framework as Ff

from FrEIA.framework import *
from FrEIA.framework import topological_order
from typing import List, Tuple, Iterable, Union, Optional
from torch import Tensor
from FrEIA.modules.base import InvertibleModule

# CNN hidden channel size, ratio to input channel size
HIDDEN_SIZE = 128

class Identity(nn.Module):
    def __init__(self, return_value=None):
        super(Identity, self).__init__()
        self.return_value = return_value
        
    def forward(self, x, *args, **kwargs):
        return x


class FeatureExtractor:

    def __init__(self, backbone):
        self.clear()
        self.bb = backbone

    def __call__(self, module, module_in, module_out):
        self.saved_feature = module_out.detach()
        if self.bb == "deit_base_distilled_patch16_384":
            self.saved_feature = self.saved_feature[:, 2:]

    def clear(self):
        self.saved_feature = None

class OwnGraphINN(InvertibleModule):

    def __init__(self, node_list, force_tuple_output=False, verbose=False):
        # Gather lists of input, output and condition nodes
        in_nodes = [node_list[i] for i in range(len(node_list))
                    if isinstance(node_list[i], InputNode)]
        out_nodes = [node_list[i] for i in range(len(node_list))
                     if isinstance(node_list[i], OutputNode)]
        condition_nodes = [node_list[i] for i in range(len(node_list)) if
                           isinstance(node_list[i], ConditionNode)]

        # Check that all nodes are in the list
        for node in node_list:
            for in_node, idx in node.inputs:
                if in_node not in node_list:
                    raise ValueError(f"{node} gets input from {in_node}, "
                                     f"but the latter is not in the node_list "
                                     f"passed to GraphINN.")
            for out_node, idx in node.outputs:
                if out_node not in node_list:
                    raise ValueError(f"{out_node} gets input from {node}, "
                                     f"but the it's not in the node_list "
                                     f"passed to GraphINN.")

        # Build the graph and tell nodes about their dimensions so that they can
        # build the modules
        node_list = topological_order(node_list, in_nodes, out_nodes)
        global_in_shapes = [node.output_dims[0] for node in in_nodes]
        global_out_shapes = [node.input_dims[0] for node in out_nodes]
        global_cond_shapes = [node.output_dims[0] for node in condition_nodes]

        # Only now we can set out shapes
        super().__init__(global_in_shapes, global_cond_shapes)
        self.node_list = node_list

        # Now we can store everything -- before calling super constructor,
        # nn.Module doesn't allow assigning anything
        self.in_nodes = in_nodes
        self.condition_nodes = condition_nodes
        self.out_nodes = out_nodes

        self.global_out_shapes = global_out_shapes
        self.force_tuple_output = force_tuple_output
        self.module_list = nn.ModuleList([n.module for n in node_list
                                          if n.module is not None])

        if verbose:
            print(self)


    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        if len(self.global_out_shapes) == 1 and not self.force_tuple_output:
            raise ValueError("You can only call output_dims on a "
                             "GraphINN with more than one output "
                             "or when setting force_tuple_output=True.")
        return self.global_out_shapes


    def forward(self, x_or_z: Union[Tensor, Iterable[Tensor]],
                c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True,
                intermediate_outputs: bool = False, x: None = None) \
            -> Tuple[Tuple[Tensor], Tensor]:
        """
        Forward or backward computation of the whole net.
        """
        if x is not None:
            x_or_z = x
            warnings.warn("You called GraphINN(x=...). x is now called x_or_z, "
                          "please pass input as positional argument.")

        if torch.is_tensor(x_or_z):
            x_or_z = x_or_z,
        if torch.is_tensor(c):
            c = c,

        jacobian = torch.zeros((x_or_z[0].shape[0], 1, *x_or_z[0].shape[2:])).to(x_or_z[0])
        outs = {}
        jacobian_dict = {} if jac else None

        # Explicitly set conditions and starts
        start_nodes = self.out_nodes if rev else self.in_nodes
        if len(x_or_z) != len(start_nodes):
            raise ValueError(f"Got {len(x_or_z)} inputs, but expected "
                             f"{len(start_nodes)}.")
        for tensor, start_node in zip(x_or_z, start_nodes):
            outs[start_node, 0] = tensor

        if c is None:
            c = []
        if len(c) != len(self.condition_nodes):
            raise ValueError(f"Got {len(c)} conditions, but expected "
                             f"{len(self.condition_nodes)}.")
        for tensor, condition_node in zip(c, self.condition_nodes):
            outs[condition_node, 0] = tensor

        # Go backwards through nodes if rev=True
        for node in self.node_list[::-1 if rev else 1]:
            # Skip all special nodes
            if node in self.in_nodes + self.out_nodes + self.condition_nodes:
                continue

            has_condition = len(node.conditions) > 0

            mod_in = []
            mod_c = []
            for prev_node, channel in (node.outputs if rev else node.inputs):
                mod_in.append(outs[prev_node, channel])
            for cond_node in node.conditions:
                mod_c.append(outs[cond_node, 0])
            mod_in = tuple(mod_in)
            mod_c = tuple(mod_c)

            try:
                if has_condition:
                    mod_out = node.module(mod_in, c=mod_c, rev=rev, jac=jac)
                else:
                    mod_out = node.module(mod_in, rev=rev, jac=jac)
            except Exception as e:
                raise RuntimeError(f"{node} encountered an error.") from e
            
            out, mod_jac = self._check_output(node, mod_out, jac, rev)

            for out_idx, out_value in enumerate(out):
                outs[node, out_idx] = out_value

            if jac:
                jacobian = jacobian + mod_jac
                jacobian_dict[node] = mod_jac

        for out_node in (self.in_nodes if rev else self.out_nodes):
            # This copies the one input of the out node
            outs[out_node, 0] = outs[(out_node.outputs if rev
                                      else out_node.inputs)[0]]

        if intermediate_outputs:
            return outs, jacobian_dict
        else:
            out_list = [outs[out_node, 0] for out_node
                        in (self.in_nodes if rev else self.out_nodes)]
            if len(out_list) == 1 and not self.force_tuple_output:
                return out_list[0], jacobian
            else:
                return tuple(out_list), jacobian


    def _check_output(self, node, mod_out, jac, rev):
        if torch.is_tensor(mod_out):
            raise ValueError(
                f"The node {node}'s module returned a tensor only. This "
                f"is deprecated without fallback. Please follow the "
                f"signature of InvertibleOperator#forward in your module "
                f"if you want to use it in a GraphINN.")

        if len(mod_out) != 2:
            raise ValueError(
                f"The node {node}'s module returned a tuple of length "
                f"{len(mod_out)}, but should return a tuple `z_or_x, jac`.")

        out, mod_jac = mod_out

        if torch.is_tensor(out):
            raise ValueError(f"The node {node}'s module returns a tensor. "
                             f"This is deprecated.")

        if len(out) != len(node.inputs if rev else node.outputs):
            raise ValueError(
                f"The node {node}'s module returned {len(out)} output "
                f"variables, but should return "
                f"{len(node.inputs if rev else node.outputs)}.")

        if not torch.is_tensor(mod_jac):
            if isinstance(mod_jac, (float, int)):
                mod_jac = torch.zeros((out[0].shape[0], 1, *out[0].shape[2:])).to(out[0].device) \
                          + mod_jac
            elif jac:
                raise ValueError(
                    f"The node {node}'s module returned a non-tensor as "
                    f"Jacobian: {mod_jac}")
            elif not jac and mod_jac is not None:
                raise ValueError(
                    f"The node {node}'s module returned neither None nor a "
                    f"Jacobian: {mod_jac}")
        return out, mod_jac

    def log_jacobian_numerical(self, x, c=None, rev=False, h=1e-04):
        """
        Approximate log Jacobian determinant via finite differences.
        """
        if isinstance(x, (list, tuple)):
            batch_size = x[0].shape[0]
            ndim_x_separate = [np.prod(x_i.shape[1:]) for x_i in x]
            ndim_x_total = sum(ndim_x_separate)
            x_flat = torch.cat([x_i.view(batch_size, -1) for x_i in x], dim=1)
        else:
            batch_size = x.shape[0]
            ndim_x_total = np.prod(x.shape[1:])
            x_flat = x.reshape(batch_size, -1)

        J_num = torch.zeros(batch_size, ndim_x_total, ndim_x_total)
        for i in range(ndim_x_total):
            offset = x[0].new_zeros(batch_size, ndim_x_total)
            offset[:, i] = h
            if isinstance(x, (list, tuple)):
                x_upper = torch.split(x_flat + offset, ndim_x_separate, dim=1)
                x_upper = [x_upper[i].view(*x[i].shape) for i in range(len(x))]
                x_lower = torch.split(x_flat - offset, ndim_x_separate, dim=1)
                x_lower = [x_lower[i].view(*x[i].shape) for i in range(len(x))]
            else:
                x_upper = (x_flat + offset).view(*x.shape)
                x_lower = (x_flat - offset).view(*x.shape)
            y_upper, _ = self.forward(x_upper, c=c, rev=rev, jac=False)
            y_lower, _ = self.forward(x_lower, c=c, rev=rev, jac=False)
            if isinstance(y_upper, (list, tuple)):
                y_upper = torch.cat(
                    [y_i.view(batch_size, -1) for y_i in y_upper], dim=1)
                y_lower = torch.cat(
                    [y_i.view(batch_size, -1) for y_i in y_lower], dim=1)
            J_num[:, :, i] = (y_upper - y_lower).view(batch_size, -1) / (2 * h)
        logdet_num = x[0].new_zeros(batch_size)
        for i in range(batch_size):
            logdet_num[i] = torch.slogdet(J_num[i])[1]

        return logdet_num


    def get_node_by_name(self, name) -> Optional[Node]:
        """
        Return the first node in the graph with the provided name.
        """
        for node in self.node_list:
            if node.name == name:
                return node
        return None


    def get_module_by_name(self, name) -> Optional[nn.Module]:
        """
        Return module of the first node in the graph with the provided name.
        """
        node = self.get_node_by_name(name)
        try:
            return node.module
        except AttributeError:
            return None

class OwnActNorm(InvertibleModule):

    def __init__(self, dims_in, dims_c=None, init_data=None):
        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in[0]
        param_dims = [1, self.dims_in[0]] + [1 for i in range(len(self.dims_in) - 1)]
        self.scale = nn.Parameter(torch.zeros(*param_dims))
        self.bias = nn.Parameter(torch.zeros(*param_dims))

        if init_data:
            self.initialize_with_data(init_data)
        else:
            self.init_on_next_batch = True

        def on_load_state_dict(*args):
            # when this module is loading state dict, we SHOULDN'T init with data,
            # because that will reset the trained parameters. Registering a hook
            # that disable this initialisation.
            self.init_on_next_batch = False
        self._register_load_state_dict_pre_hook(on_load_state_dict)

    def initialize_with_data(self, data):
        # Initialize to mean 0 and std 1 with sample batch
        # 'data' expected to be of shape (batch, channels[, ...])
        assert all([data.shape[i+1] == self.dims_in[i] for i in range(len(self.dims_in))]),\
            "Can't initialize ActNorm layer, provided data don't match input dimensions."
        self.scale.data.view(-1)[:] \
            = torch.log(1 / data.transpose(0,1).contiguous().view(self.dims_in[0], -1).std(dim=-1))
        data = data * self.scale.exp()
        self.bias.data.view(-1)[:] \
            = -data.transpose(0,1).contiguous().view(self.dims_in[0], -1).mean(dim=-1)
        self.init_on_next_batch = False

    def forward(self, x, rev=False, jac=True):
        if self.init_on_next_batch:
            self.initialize_with_data(x[0])

        #jac = (self.scale.sum() * np.prod(self.dims_in[1:])).repeat(x[0].shape[0])
        jac = self.scale.sum(dim=1, keepdim=True).repeat(x[0].shape[0], 1, *self.dims_in[1:])
        if rev:
            jac = -jac

        if not rev:
            return [x[0] * self.scale.exp() + self.bias], jac
        else:
            return [(x[0] - self.bias) / self.scale.exp()], jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class FastFlowBlock(Fm.coupling_layers.GLOWCouplingBlock):

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=0.15, clamp_activation="ATAN"):
        super().__init__(dims_in, dims_c=dims_c, subnet_constructor=subnet_constructor, clamp=clamp, clamp_activation=clamp_activation)
        
        self.subnet1.apply(init_with_xavier)
        self.subnet2.apply(init_with_xavier)

    def _coupling1(self, x1, u2, rev=False):
        a2 = self.subnet2(u2)
        s2, t2 = a2[:, :self.split_len1], a2[:, self.split_len1:]
        s2 = self.clamp * self.f_clamp(s2)
        j1 = s2

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = a1[:, :self.split_len2], a1[:, self.split_len2:]
        s1 = self.clamp * self.f_clamp(s1)
        j2 = s1

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, j2

def subnet_conv_3x3(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, HIDDEN_SIZE, 3, padding=1), nn.ReLU(),
        nn.Conv2d(HIDDEN_SIZE, c_out, 3, padding=1))

def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, HIDDEN_SIZE, 1), nn.ReLU(),
        nn.Conv2d(HIDDEN_SIZE, c_out, 1))

def init_with_xavier(module):
    #gain = nn.init.calculate_gain('relu')
    gain = 1/50.0
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight, gain=gain)

def init_last_conv_with_zeros(module):
    if isinstance(module[-1], nn.Conv2d):
        nn.init.zeros_(module[-1].weight)
        nn.init.zeros_(module[-1].bias)

def build_fast_flow(clamp, clamp_activation, encoded_shape=(768, 28, 28)):
    nodes = [Ff.InputNode(*encoded_shape, name='Input')]
    for i in range(20):
        nodes.append(Ff.Node(nodes[-1], OwnActNorm, {}, name='ActNorm'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {}, name="ChannelPermute"))
        if i % 2 == 0:
            nodes.append(Ff.Node(
                nodes[-1],
                FastFlowBlock,
                {
                    'subnet_constructor': subnet_conv_3x3,
                    'clamp': clamp,
                    'clamp_activation': clamp_activation
                },
                name='FastFlowStep_{}_3x3'.format(i)
            ))
        else:
            nodes.append(Ff.Node(
                nodes[-1],
                FastFlowBlock,
                {
                    'subnet_constructor': subnet_conv_1x1,
                    'clamp': clamp,
                    'clamp_activation': clamp_activation
                },
                name='FastFlowStep_{}_1x1'.format(i)
            ))
    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    conv_inn = OwnGraphINN(nodes)
    return conv_inn