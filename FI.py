import sys
import copy
import random 
import numpy as np
from enum import Enum
from datatype import floatToBinary32, binaryToFloat32, int8ToBinary, binaryToint8
import torch
# testconversion()
# supported fault types
class FaultTypes(Enum):
    EXPONENT_RANDOM = "ExponentRandom"
    MANTISSA_RANDOM = "MantissaRandom"
    SIGN = "Sign"
    EXPONENT_FIXED = "ExponentFixed"
    MANTISSA_FIXED = "MantissaFixed"
# end of fault types

    
def bitFlip(bin_value, bit, float_type):
    ''' flip one bit in a 64-bit signed integer'''
    "The input and output are both binary representations (string representation)"
    assert type(bin_value) == str
    assert type(bit) == int
    assert bit <= 63 and bit >=0
    # for IEEE 754 single precision floating point numbers
    # if 'Double' in float_type:
    #     assert len(bin_value) == 64
    #     assert bit <= 63 and bit >=0
    #     if bit == 62:
    #         print("Flipping Sign bit")
    #     elif bit <= 50:
    #         print("Flipping Mantissa bit")
    #     else:
    #         print("Flipping Exponent bit")
    # elif 'Float' in float_type:
    #     assert len(bin_value) == 32
    #     assert bit <= 31 and bit >=0
    #     if bit == 31:
    #         print("Flipping Sign bit")
    #     elif bit <= 22:
    #         print("Flipping Mantissa bit")
    #     else:
    #         print("Flipping Exponent bit")

    # reverse binary string because the bit index count from left to right, while the string index count from right to left.
    bin_value = bin_value[::-1]

    # flip bit 
    if bin_value[bit] == "1":
        bin_value =  bin_value[:bit] + "0" + bin_value[bit+1:]
    else:
        bin_value =  bin_value[:bit] + "1" + bin_value[bit+1:]

    # reverse sting back 
    return bin_value[::-1]


def faultInjection(model_params_new, fault_type, is_compressed, is_quantized, layer_name):
    '''Inject single fault to model parameters and return the parameters with fault injected'''
    # model_params_new = copy.deepcopy(model_params)
    quant_bits = 0

    if is_quantized:
        quant_bits = 8
        inject_bit = 6 # the highest bit except the sign bit 
    else:
        inject_bit = 30
        

    # select parameter to inject fault 
    if layer_name == None:
        while True:
            layer_key, layer_params = random.choice(list(model_params_new.items()))
            dimension = len(layer_params.shape)
            if ("weight" in layer_key):
                print ('dimension', dimension)
                if (dimension == 4) or (dimension == 2):
                # if dimension == 4:
                    break
    else:
        for name, value in model_params_new.items():
            if (layer_name in name) and 'weight' in name:
                layer_key, layer_params = name, value
                dimension = len(layer_params.shape)

    while True:
        # if quantized, the selected value is INT8
        # if not quantized, the selected value is FP32
        rand_indexes = np.floor(np.random.rand(dimension) * layer_params.shape).astype(int)

        if dimension == 4:
            selected = layer_params[rand_indexes[0], rand_indexes[1], rand_indexes[2], rand_indexes[3]].cpu()
        elif dimension == 3:
            selected = layer_params[rand_indexes[0], rand_indexes[1], rand_indexes[2]].cpu()
        elif dimension == 2:
            selected = layer_params[rand_indexes[0],rand_indexes[1]].cpu()
        else:
            selected = layer_params[rand_indexes[0]]
        if (is_compressed == False) or (torch.abs(selected) > 1e-7):
            break
    
    # flip one bit 
    if is_quantized == False:
        if fault_type == FaultTypes.EXPONENT_FIXED: 
            print("Flipping...")
            converted = floatToBinary32(selected)
            flipped = bitFlip(converted, inject_bit, selected.type())
            print ("Binary Value before flip and after flip", converted, flipped)
            flipped = binaryToFloat32(flipped)
            print ("Decimal Value before flip and after flip", selected, flipped)

        elif fault_type == FaultTypes.EXPONENT_RANDOM:
            flipped = floatToBinary32(bitFlip(floatToBinary32(selected), random.randint(23, 31), selected.type()))
        elif fault_type == FaultTypes.MANTISSA_FIXED:
            flipped = floatToBinary32(bitFlip(floatToBinary32(selected), 22, selected.type()))
        elif fault_type == FaultTypes.MANTISSA_RANDOM:
            flipped = floatToBinary32(bitFlip(floatToBinary32(selected), random.randint(0, 23), selected.type()))
        elif fault_type == FaultTypes.Sign:
            flipped = floatToBinary32(bitFlip(floatToBinary32(selected), 31, selected.type()))
        else:
            pass 
    else:# for quantized value
        if dimension == 1:
            # TODO: support bias injetions
            # print ("biases: ", layer_params)
            # model_params_new[layer_key].requires_grad = False

            b_scale, b_zero_point = calculate_bias_and_zero_point(layer_params)
            selected = int(torch.round(selected/b_scale + b_zero_point))

            converted = int8ToBinary(selected)
            flipped = bitFlip(converted, inject_bit, torch.int8)
            print ("Value before flipped and after flipped", converted, flipped)
            flipped = binaryToint8(flipped)

        elif (dimension == 4) or (dimension == 2):
            # print ("=====selected dtype: ", selected.dtype())
            converted = int8ToBinary(selected.int_repr())
            flipped = bitFlip(converted, inject_bit, torch.int8)
            print ("Value before flipped and after flipped", converted, flipped)
            flipped = binaryToint8(flipped)
        else:
            pass
    
    # convert flipped integer to quantized representation (stored as float, scale and zero point)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_quantized:
        if dimension == 4 or dimension == 2:
            flipped = (flipped - selected.q_zero_point())*selected.q_scale()
        elif dimension == 1:
            flipped = (flipped - b_zero_point)*b_scale
        else:
            pass
    else:
        pass

    # modify the selected parameter
    if dimension == 4:
        model_params_new[layer_key][rand_indexes[0], rand_indexes[1], rand_indexes[2], rand_indexes[3]] = flipped
    elif dimension == 3:
        model_params_new[layer_key][rand_indexes[0], rand_indexes[1], rand_indexes[2]] = flipped
    elif dimension == 2:
        model_params_new[layer_key][rand_indexes[0],rand_indexes[1]] = flipped
    else:
        model_params_new[layer_key][rand_indexes[0]] = flipped


    # print ("modified parameter", model_params_new[layer_key][rand_indexes[0], rand_indexes[1], rand_indexes[2], rand_indexes[3]])
    return model_params_new, layer_key, rand_indexes


def calculate_bias_and_zero_point(x):
    # minMax quantization 
    # TODO
    qmin, qmax = -128, 127 # int8
    eps = torch.finfo(torch.float32).eps
    min_val = torch.min(x)
    max_val = torch.max(x)
    min_val = min(0.0, min_val)
    max_val = max(0.0, max_val)
    if max_val == min_val:
        scale = 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / float(qmax - qmin)
        scale = max(scale, eps)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        zero_point = int(zero_point)

    return scale, zero_point

