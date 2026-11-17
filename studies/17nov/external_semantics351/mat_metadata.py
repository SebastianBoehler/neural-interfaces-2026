"""Read Scherer's existing MAT-v5 metadata, skipping the EEG numeric payload."""
import struct
import zlib

import numpy as np

DTYPES = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4',
          7: '<f4', 9: '<f8', 12: '<i8', 13: '<u8', 16: 'u1', 17: '<u2'}


def tag(buf, offset):
    kind, size = struct.unpack_from('<II', buf, offset)
    if kind >> 16:
        return kind & 65535, buf[offset + 4:offset + 4 + (kind >> 16)], offset + 8
    return kind, buf[offset + 8:offset + 8 + size], offset + 8 + ((size + 7) // 8) * 8


def matrix(buf, skip=False):
    kind, flags, pos = tag(buf, 0)
    assert kind == 6
    cls = struct.unpack_from('<I', flags)[0] & 255
    kind, dims, pos = tag(buf, pos)
    assert kind == 5
    shape = tuple(np.frombuffer(dims, dtype='<i4').tolist())
    kind, name, pos = tag(buf, pos)
    assert kind == 1
    if skip:
        return {'shape': shape, 'numeric_payload_skipped': True}
    count = int(np.prod(shape))
    if cls == 1:
        result = []
        for _ in range(count):
            kind, value, pos = tag(buf, pos)
            assert kind == 14
            result.append(matrix(value))
        return result
    if cls == 2:
        kind, width, pos = tag(buf, pos)
        assert kind == 5 and count == 1
        width = struct.unpack_from('<i', width)[0]
        kind, names, pos = tag(buf, pos)
        assert kind == 1
        fields = [bytes(names[i:i + width]).split(b'\0')[0].decode()
                  for i in range(0, len(names), width)]
        result = {}
        for field in fields:
            kind, value, pos = tag(buf, pos)
            assert kind == 14
            result[field] = matrix(value, skip=field == 'X')
        return result
    kind, data, pos = tag(buf, pos)
    assert kind in DTYPES, (cls, kind)
    values = np.frombuffer(data, dtype=DTYPES[kind])
    if cls == 4:
        return ''.join(chr(int(v)) for v in values)
    assert cls in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    return values.reshape(shape, order='F').squeeze().copy()


def read(path):
    blob = path.read_bytes()
    assert blob[126:128] == b'IM'
    kind, data, _ = tag(memoryview(blob), 128)
    assert kind == 15, 'Expected pinned compressed MAT-v5 container'
    # Container decompression is necessary; X is never converted to a numeric EEG array.
    unpacked = zlib.decompress(data)
    kind, value, _ = tag(memoryview(unpacked), 0)
    assert kind == 14
    return matrix(value)
