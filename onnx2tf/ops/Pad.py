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
    paddings = tf.cast(paddings, dtype=tf.int32)

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
        tf_layers_dict[graph_node_output.name]['tf_node'] = tf.pad(
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
            empty_paddings = np.zeros([tensor_rank, 2], dtype=np.int32)
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
                    input_tensor_padded = tf.pad(
                        input_tensor_padded, temp_empty_paddings, 'SYMMETRIC'
                    )
                # end
                end_loop_count = empty_paddings[idx][1]
                temp_empty_paddings = copy.deepcopy(empty_paddings)
                temp_empty_paddings[idx][0] = 0
                temp_empty_paddings[idx][1] = 1
                for _ in range(end_loop_count):
                    input_tensor_padded = tf.pad(
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
