#!/usr/bin/env python3
"""Convert ds4 MTP GGUF to standard deepseek4 GGUF for Fringe210 -md flag."""
import sys, os, numpy as np
from collections import OrderedDict
from typing import Any

fringe_py = "/tmp/fringe-analysis/gguf-py"
if fringe_py not in sys.path:
    sys.path.insert(0, fringe_py)

from gguf import GGUFReader, GGUFValueType
from gguf.gguf_writer import GGUFWriter


def extract_value(field) -> Any:
    types = field.types
    if not types:
        return None
    main_type = types[0]
    if main_type == GGUFValueType.STRING:
        return str(bytes(field.parts[field.data[0]]), encoding='utf-8')
    if main_type == GGUFValueType.ARRAY:
        if len(types) < 2:
            return None
        sub_type = types[1]
        arr = field.parts[field.data[0]]
        if sub_type == GGUFValueType.STRING:
            return [str(bytes(s), encoding='utf-8') for s in arr]
        return arr.tolist()
    val = field.parts[field.data[0]]
    if hasattr(val, '__iter__') and len(val) == 1:
        return val[0].item() if hasattr(val[0], 'item') else val[0]
    return val.item() if hasattr(val, 'item') else val


def remap_name(name: str) -> str:
    if name.startswith("mtp.0."):
        return "blk.0." + name[len("mtp.0."):]
    return name


def convert(input_path: str, output_path: str):
    print(f"Reading: {input_path}")
    reader = GGUFReader(input_path, 'r')

    metadata = OrderedDict()
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        try:
            val = extract_value(field)
            metadata[key] = (val, field.types)
        except Exception as e:
            print(f"  WARNING: Could not extract key {key}: {e}")

    arch = "deepseek4"
    print(f"Target arch: {arch}")
    print(f"Tensors: {len(reader.tensors)}")
    for t in reader.tensors[:5]:
        print(f"  {t.name} shape={list(t.shape)} type={t.tensor_type.name} nbytes={t.n_bytes}")
    if len(reader.tensors) > 5:
        print(f"  ... and {len(reader.tensors) - 5} more")

    writer = GGUFWriter(output_path, arch)

    for key, (val, types) in metadata.items():
        if key == "general.architecture":
            continue
        main_type = types[0]
        try:
            if main_type == GGUFValueType.STRING:
                writer.add_string(key, val)
            elif main_type == GGUFValueType.UINT32:
                writer.add_uint32(key, int(val))
            elif main_type == GGUFValueType.UINT64:
                writer.add_uint64(key, int(val))
            elif main_type == GGUFValueType.INT32:
                writer.add_int32(key, int(val))
            elif main_type == GGUFValueType.INT64:
                writer.add_int64(key, int(val))
            elif main_type == GGUFValueType.FLOAT32:
                writer.add_float32(key, float(val))
            elif main_type == GGUFValueType.FLOAT64:
                writer.add_float64(key, float(val))
            elif main_type == GGUFValueType.BOOL:
                writer.add_bool(key, bool(val))
            elif main_type == GGUFValueType.ARRAY:
                writer.add_array(key, val)
            elif main_type == GGUFValueType.UINT8:
                writer.add_uint8(key, int(val))
            elif main_type == GGUFValueType.UINT16:
                writer.add_uint16(key, int(val))
            elif main_type == GGUFValueType.INT8:
                writer.add_int8(key, int(val))
            elif main_type == GGUFValueType.INT16:
                writer.add_int16(key, int(val))
            else:
                print(f"  SKIP: {key} type={main_type}")
        except Exception as e:
            print(f"  ERROR adding key {key}: {e}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    print(f"\nWriting tensors with remapped names...")
    for i, rt in enumerate(reader.tensors):
        new_name = remap_name(rt.name)
        if new_name != rt.name:
            print(f"  [{i}] {rt.name} -> {new_name}")
        writer.add_tensor(new_name, rt.data, raw_dtype=rt.tensor_type)

    writer.write_tensors_to_file(progress=True)
    print(f"\nDone: {output_path}")
    for key in metadata:
        if any(kw in key for kw in ['block_count','head_count','layer_count','expert_count','n_hc']):
            print(f"  {key}: {metadata[key][0]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_mtp_gguf.py <input.mtp.gguf> [output.gguf]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "mtp-converted-deepseek4.gguf"
    if not os.path.exists(inp):
        print(f"Error: {inp} not found")
        sys.exit(1)
    convert(inp, out)
