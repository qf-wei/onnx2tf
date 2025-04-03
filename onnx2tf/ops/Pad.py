import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    stridedslice_with_flexing_deterrence,
)

def custom_pad(tensor, paddings, mode="CONSTANT", constant_value=0.0):
    """
    Applies padding to a float32 tensor using only basic TF ops, supporting:
      - "CONSTANT": pads with a constant value.
      - "REFLECT": pads with a reflection of the tensor (excludes the border element).
      - "SYMMETRIC": pads with a symmetric reflection (includes the border element).
    
    Parameters
    ----------
    tensor : tf.Tensor
        A float32 tensor to pad.
    paddings : array-like or tf.Tensor
        A tensor or array of shape [rank, 2] (type int32) where each row is [pad_before, pad_after].
    mode : str, optional
        One of "CONSTANT", "REFLECT", or "SYMMETRIC". (Default is "CONSTANT")
    constant_value : float, optional
        The constant value to pad with when mode is "CONSTANT". (Default is 0.0)
        
    Returns
    -------
    tf.Tensor
        The padded tensor (float32).
        
    Note
    ----
    This function manually implements the pad behavior without calling tf.pad so that
    the resulting graph uses float32 everywhere (which may ease Core ML conversion).
    """
    # Ensure the input tensor is float32.
    tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    
    # Ensure paddings is a tensor of type int32.
    if not isinstance(paddings, tf.Tensor):
        paddings = tf.convert_to_tensor(paddings, dtype=tf.int32)
    else:
        paddings = tf.cast(paddings, tf.int32)
    
    # Try to get a static rank.
    rank = tensor.shape.rank
    if rank is None:
        raise ValueError("The input tensor must have a statically-known rank.")
    
    # For simplicity, we assume paddings are known statically.
    # (If they aren’t, you might add dynamic logic using tf.shape and tf.unstack.)
    paddings_np = paddings.numpy() if hasattr(paddings, "numpy") else np.array(paddings)
    
    padded_tensor = tensor
    # Loop over each dimension, applying padding one dimension at a time.
    for dim in range(rank):
        pad_before = int(paddings_np[dim, 0])
        pad_after  = int(paddings_np[dim, 1])
        current_shape = padded_tensor.shape.as_list()
        
        if mode.upper() == "CONSTANT":
            # Build padding slices filled with constant_value.
            before_shape = current_shape.copy()
            before_shape[dim] = pad_before
            before_pad = tf.fill(before_shape, constant_value)
            after_shape = current_shape.copy()
            after_shape[dim] = pad_after
            after_pad = tf.fill(after_shape, constant_value)
        elif mode.upper() == "REFLECT":
            # REFLECT mode: reflect without repeating the border.
            # (Requires pad_before, pad_after <= current dimension size - 1)
            if pad_before > 0:
                # Slice from index 1 to 1+pad_before and reverse along dim.
                begin = [0] * rank
                begin[dim] = 1
                size = current_shape.copy()
                size[dim] = pad_before
                before_slice = tf.slice(padded_tensor, begin, size)
                before_pad = tf.reverse(before_slice, axis=[dim])
            else:
                before_pad = None
            if pad_after > 0:
                begin = [0] * rank
                # Start from (current_dim - pad_after - 1) to exclude the edge element.
                begin[dim] = current_shape[dim] - pad_after - 1
                size = current_shape.copy()
                size[dim] = pad_after
                after_slice = tf.slice(padded_tensor, begin, size)
                after_pad = tf.reverse(after_slice, axis=[dim])
            else:
                after_pad = None
        elif mode.upper() == "SYMMETRIC":
            # SYMMETRIC mode: reflect including the border.
            if pad_before > 0:
                begin = [0] * rank
                begin[dim] = 0
                size = current_shape.copy()
                size[dim] = pad_before
                before_slice = tf.slice(padded_tensor, begin, size)
                before_pad = tf.reverse(before_slice, axis=[dim])
            else:
                before_pad = None
            if pad_after > 0:
                begin = [0] * rank
                begin[dim] = current_shape[dim] - pad_after
                size = current_shape.copy()
                size[dim] = pad_after
                after_slice = tf.slice(padded_tensor, begin, size)
                after_pad = tf.reverse(after_slice, axis=[dim])
            else:
                after_pad = None
        else:
            raise ValueError("Unsupported pad mode: {}".format(mode))
        
        # Concatenate along the current dimension.
        if mode.upper() == "CONSTANT":
            padded_tensor = tf.concat([before_pad, padded_tensor, after_pad], axis=dim)
        else:
            concat_list = []
            if before_pad is not None:
                concat_list.append(before_pad)
            concat_list.append(padded_tensor)
            if after_pad is not None:
                concat_list.append(after_pad)
            padded_tensor = tf.concat(concat_list, axis=dim)
    
    return padded_tensor




@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Pad

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans = True
    if len(graph_node.inputs) == 1:
        before_op_output_shape_trans_1 = \
            tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = before_op_output_shape_trans_1
    elif len(graph_node.inputs) >= 2:
        before_op_output_shape_trans_1 = \
            tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans_2 = \
            tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
        before_op_output_shape_trans = (
            before_op_output_shape_trans_1 and before_op_output_shape_trans_2
        )

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )

    constant_value = 0
    if len(graph_node.inputs) >= 3 and graph_node.inputs[2].name != '':
        constant_value = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Resolve the actual tensors
    input_tensor = (
        tf_layers_dict[graph_node_input.name]['tf_node']
        if isinstance(graph_node_input, gs.Variable)
        else graph_node_input
    )

    tensor_rank = len(input_tensor.shape)

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    constant_value = (
        tf_layers_dict[constant_value.name]['tf_node']
        if isinstance(constant_value, gs.Variable)
        else constant_value
    )

    # Grab paddings
    paddings = None
    if len(graph_node.inputs) >= 2:
        paddings_var = get_constant_or_variable(
            graph_node.inputs[1],
            False,
        )
        paddings = (
            tf_layers_dict[paddings_var.name]['tf_node']
            if isinstance(paddings_var, gs.Variable)
            else paddings_var
        )

    paddings = graph_node.attrs.get('pads', paddings)
    if isinstance(paddings, list):
        paddings = np.asarray(paddings)

    values = None
    if hasattr(paddings, 'values'):
        # Keras tensor with .values
        values = paddings.values
    elif isinstance(paddings, np.ndarray):
        # NumPy
        values = paddings
    elif hasattr(paddings, 'numpy'):
        # Eager Tensor
        values = paddings.numpy()

    # Convert ONNX-style pad to TF-style pad if it's a list or array
    if values is not None:
        # If shape is [2*tensor_rank]
        # we reshape to [2, tensor_rank] then transpose
        paddings = values.reshape([2, tensor_rank]).transpose()
        paddings_rank = paddings.shape[0]
        if paddings_rank > 2:
            # Re-interpret ONNX [begin0, begin1, ..., end0, end1,...] style
            paddings = np.asarray(
                [
                    [begin, end]
                    for begin, end in zip(
                        values[0:tensor_rank], values[tensor_rank:tensor_rank + tensor_rank]
                    )
                ],
                dtype=values.dtype,
            )
            if before_op_output_shape_trans:
                convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
                new_values = [0] * tensor_rank
                for new_idx, idx in enumerate(convertion_table):
                    new_values[new_idx] = paddings[idx]
                paddings = np.asarray(new_values, dtype=paddings.dtype)
            paddings = (
                tf.convert_to_tensor(paddings)
                if isinstance(paddings, np.ndarray)
                else paddings
            )
    elif tf_keras.backend.is_keras_tensor(paddings):
        # Possibly shape [2, rank], transpose that
        paddings = tf.transpose(
            a=tf.reshape(paddings, shape=[2, tensor_rank])
        )
        paddings_rank = paddings.shape[0]
        if paddings_rank > 2:
            arr = []
            for begin, end in paddings:
                arr.append([begin, end])
            paddings = arr
            if before_op_output_shape_trans:
                convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
                new_values = [0] * tensor_rank
                for new_idx, idx in enumerate(convertion_table):
                    new_values[new_idx] = paddings[idx]
                paddings = new_values

    mode = graph_node.attrs.get('mode', 'constant')

    # Preserve Graph Structure
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': (
            tf_layers_dict[graph_node_input.name]['nhwc']
            if (
                isinstance(graph_node_input, gs.Variable)
                and 'nhwc' in tf_layers_dict[graph_node_input.name]
            )
            else False
        ),
    }

    # Ensure paddings is int32
    if isinstance(paddings, np.ndarray):
        paddings = tf.convert_to_tensor(paddings)
    paddings = tf.cast(paddings, dtype=tf.float32)

    # Negative number padding workaround
    if (
        input_tensor.shape != tf.TensorShape(None)
        and None not in input_tensor.shape
        and hasattr(paddings, 'numpy')
        and (paddings.numpy() < 0).any()
    ):
        begin_ = [
            -1 if padding[0] >= 0 else -padding[0].numpy() for padding in paddings
        ]
        begin_mask_ = tf.convert_to_tensor(
            sum([2**idx if begin == -1 else 0 for idx, begin in enumerate(begin_)])
        )
        end_ = [
            -1 if padding[1] >= 0 else -padding[1].numpy() for padding in paddings
        ]
        end_mask_ = tf.convert_to_tensor(
            sum([2**idx if end == -1 else 0 for idx, end in enumerate(end_)])
        )
        begin_ = tf.convert_to_tensor(
            [0 if val == -1 else val for val in begin_]
        )
        end_ = [
            0 if val == -1 else input_tensor.shape[idx] - val
            for idx, val in enumerate(end_)
        ]
        end_ = tf.convert_to_tensor(end_)
        strides_ = None

        COMPRESSION_DEFAULT_VALUE = 5
        input_tensor_rank = len(input_tensor.shape)
        if input_tensor_rank > COMPRESSION_DEFAULT_VALUE:
            ignore_axes = [idx for idx in range(input_tensor_rank)]
            input_tensor = stridedslice_with_flexing_deterrence(
                input_tensor=input_tensor,
                begin=begin_,
                end=end_,
                strides=strides_,
                begin_mask=begin_mask_,
                end_mask=end_mask_,
                ignore_axes=ignore_axes,
                compression_defult_value=COMPRESSION_DEFAULT_VALUE,
                onnx_slice_dims_count=input_tensor_rank,
                **kwargs,
            )
        else:
            input_tensor = tf.strided_slice(
                input_=input_tensor,
                begin=begin_,
                end=end_,
                strides=strides_,
                begin_mask=begin_mask_,
                end_mask=end_mask_,
            )
        paddings_numpy: np.ndarray = paddings.numpy()
        paddings_numpy[paddings_numpy < 0] = 0
        paddings = tf.convert_to_tensor(paddings_numpy)

    # ------------------------------------------------------------------------------
    # IMPORTANT FIX: Cast the input to float if it's integer
    # ------------------------------------------------------------------------------
    if isinstance(input_tensor, tf.Tensor) and input_tensor.dtype.is_integer:
        input_tensor = tf.cast(input_tensor, tf.float32)
    # If constant_value is also integer, cast it to match
    if isinstance(constant_value, tf.Tensor) and constant_value.dtype.is_integer:
        constant_value = tf.cast(constant_value, tf.float32)
    elif isinstance(constant_value, int):
        # normal Python int -> cast to float
        constant_value = float(constant_value)
    # ------------------------------------------------------------------------------

    # Create the TF pad op
    if mode != 'edge':
        # mode != 'edge'
        tf_layers_dict[graph_node_output.name]['tf_node'] = custom_pad(
            tensor=input_tensor,
            paddings=paddings,
            mode=mode,
            constant_values=constant_value,
            name=graph_node.name,
        )
    else:
        # mode = 'edge' (a.k.a. 'SYMMETRIC')
        input_tensor_padded = input_tensor
        for idx, p in enumerate(paddings):
            begin_, end_ = p[0], p[1]
            empty_paddings = np.zeros([tensor_rank, 2], dtype=np.float32)
            for idxe, empty_padding in enumerate(empty_paddings):
                if idxe == idx:
                    empty_padding[0], empty_padding[1] = begin_, end_
            if not (empty_paddings == 0).all():
                # begin
                begin_loop_count = empty_paddings[idx][0]
                temp_empty_paddings = copy.deepcopy(empty_paddings)
                temp_empty_paddings[idx][0] = 1
                temp_empty_paddings[idx][1] = 0
                for _ in range(begin_loop_count):
                    input_tensor_padded = custom_pad(
                        input_tensor_padded, temp_empty_paddings, 'SYMMETRIC'
                    )
                # end
                end_loop_count = empty_paddings[idx][1]
                temp_empty_paddings = copy.deepcopy(empty_paddings)
                temp_empty_paddings[idx][0] = 0
                temp_empty_paddings[idx][1] = 1
                for _ in range(end_loop_count):
                    input_tensor_padded = custom_pad(
                        input_tensor_padded, temp_empty_paddings, 'SYMMETRIC'
                    )
        tf_layers_dict[graph_node_output.name]['tf_node'] = input_tensor_padded

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = make_tf_node_info(
        node_info={
            'tf_op_type': 'Pad',
            'tf_inputs': {
                'x': input_tensor,
                'paddings': paddings,
                'constant_value': constant_value,
                'mode': mode,
                'tensor_rank': tensor_rank,
            },
            'tf_outputs': {
                'output': tf_layers_dict[graph_node_output.name]['tf_node'],
            },
        }
    )
