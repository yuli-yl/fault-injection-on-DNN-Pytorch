import struct
# from FI import bitFlip
# reference: https://www.h-schmidt.net/FloatConverter/IEEE754.html

# remove the "ob" character in the binary representation.
getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]


def floatToBinary32(value):
    val = struct.unpack('I', struct.pack('f', value))[0]
    binvar = getBin(val)
    while value > 0 and (len(binvar) < 32): 
        binvar =  '0' + binvar
    return binvar

def binaryToFloat32(value):
    hx = hex(int(value, 2))   
    return struct.unpack("f", struct.pack("I", int(hx, 16)))[0]


def int8ToBinary(value):
    if value >= 0:
        binvar = '{0:08b}'.format(value)
    else:
        # two's complement representation
        value = 128 + value
        binvar = '1'+'{0:07b}'.format(value)
    return binvar


def binaryToint8(value):
    if value[0] == '0':
        it = int(value, 2)
    elif value[0] == '1':
        new_val = value[1:]
        it = int(new_val, 2) -128
    return it


def testconversion():
    # 64-bit conversion
    x = 0.00025
    binstr = floatToBinary32(x)
    print('Binary equivalent of ', x)
    print(binstr + '\n')
    # binstr = bitFlip(binstr, 30, 'Float')
    fl = binaryToFloat32(binstr)
    print('Decimal equivalent of Flipped  ' + binstr)
    print(fl)

    print ('\n')
    binstr = floatToBinary32(-x)

    print('Binary equivalent of :', -x)
    print(binstr + '\n')
    # binstr = bitFlip(binstr, 30, 'Float')

    fl = binaryToFloat32(binstr)
    print('Decimal equivalent of flipped ' + binstr)
    print(fl)


def test_int8():
    x = 8
    binstr = int8ToBinary(x)
    print('Binary equivalent of ', x)
    print(binstr + '\n')

    it = binaryToint8(binstr)
    print('Decimal equivalent of Flipped  ' + binstr)
    print(it)

# testconversion()
# test_int8()