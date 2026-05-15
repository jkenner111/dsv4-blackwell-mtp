#!/usr/bin/env python3
"""Convert ds4 MTP GGUF to standalone deepseek4 draft GGUF for Fringe210 -md flag.

Reads MTP GGUF, seeds metadata+tensors from donor main model, overlays MTP keys,
renames tensors mtp.0.* -> blk.0.*, forces block_count=1 and hash_layer_count=0.

Usage:
  python3 convert_mtp_gguf.py <mtp.gguf> --donor <main.gguf> [output.gguf]

Patches: P1-P8 (see git log)
"""
import sys, os, numpy as np
from collections import OrderedDict
from typing import Any, Sequence

for candidate in [
    os.path.expanduser("~/llama-dsv4-fringe/gguf-py"),
    "/tmp/fringe-analysis/gguf-py",
]:
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)
        break

try:
    from gguf import GGUFReader, GGUFValueType
    from gguf.gguf_writer import GGUFWriter
except ImportError:
    print("Error: gguf-py not found")
    sys.exit(1)

DONOR_TENSORS = [
    "token_embd.weight",
    "output_norm.weight",
    "output.weight",
    "output_hc_base.weight",
    "output_hc_fn.weight",
    "output_hc_scale.weight",
]


def read_metadata(reader, label):
    md = OrderedDict()
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        try:
            md[key] = (field.contents(), field.types)
        except Exception as e:
            print(f"  WARNING [{label}]: key {key}: {e}")
    return md


def write_metadata(writer, metadata):
    for key, (val, types) in metadata.items():
        if key == "general.architecture":
            continue
        mt = types[0]
        try:
            if mt == GGUFValueType.STRING:   writer.add_string(key, val)
            elif mt == GGUFValueType.UINT32:  writer.add_uint32(key, int(val))
            elif mt == GGUFValueType.UINT64:  writer.add_uint64(key, int(val))
            elif mt == GGUFValueType.INT32:   writer.add_int32(key, int(val))
            elif mt == GGUFValueType.INT64:   writer.add_int64(key, int(val))
            elif mt == GGUFValueType.FLOAT32: writer.add_float32(key, float(val))
            elif mt == GGUFValueType.FLOAT64: writer.add_float64(key, float(val))
            elif mt == GGUFValueType.BOOL:    writer.add_bool(key, bool(val))
            elif mt == GGUFValueType.ARRAY:   writer.add_array(key, val)
            elif mt == GGUFValueType.UINT8:   writer.add_uint8(key, int(val))
            elif mt == GGUFValueType.UINT16:  writer.add_uint16(key, int(val))
            elif mt == GGUFValueType.INT8:    writer.add_int8(key, int(val))
            elif mt == GGUFValueType.INT16:   writer.add_int16(key, int(val))
            else: print(f"  SKIP: {key} type={mt}")
        except Exception as e:
            print(f"  ERROR: {key}: {e}")


def slice_arrays(metadata, donor_bc):
    """P6: Slice per-layer arrays to length 1 to match block_count=1."""
    for key, (val, types) in list(metadata.items()):
        if types[0] != GGUFValueType.ARRAY:
            continue
        if not isinstance(val, (list, np.ndarray)):
            continue
        n = len(val)
        if n in {donor_bc, donor_bc + 1}:
            last = val[-1]
            if hasattr(last, 'item'):
                last = last.item()
            metadata[key] = ([last], types)
            print(f"  sliced {key}: {n} -> 1 (kept last: {last})")


def force_override(merged, suffix, new_val):
    """Set a metadata key ending with suffix to new_val."""
    for key in merged:
        if key.endswith(suffix):
            merged[key] = (new_val, merged[key][1])
            print(f"Forced {key} = {new_val}")
            return


def convert(input_path: str, donor_path: str, output_path: str):
    print(f"MTP:  {input_path}")
    reader = GGUFReader(input_path, 'r')
    mtp_meta = read_metadata(reader, "mtp")
    print(f"  MTP keys: {len(mtp_meta)}")

    if not os.path.exists(donor_path):
        print("Error: --donor required"); sys.exit(1)

    print(f"Donor: {donor_path}")
    donor = GGUFReader(donor_path, 'r')
    donor_meta = read_metadata(donor, "donor")
    print(f"  Donor keys: {len(donor_meta)}")

    donor_bc = None
    for key in donor_meta:
        if key.endswith(".block_count"):
            donor_bc = int(donor_meta[key][0])
            break
    print(f"  Donor block_count: {donor_bc}")

    donor_tensors = {t.name: t for t in donor.tensors}
    for name in DONOR_TENSORS:
        if name in donor_tensors:
            t = donor_tensors[name]
            print(f"  Donor tensor: {name}  {list(t.shape)}  {t.n_bytes} bytes")
        else:
            print(f"  WARNING: {name} not in donor")

    # P5: seed from donor, overlay MTP
    merged = OrderedDict(donor_meta)
    merged.update(mtp_meta)

    # P6: slice per-layer arrays
    if donor_bc is not None:
        slice_arrays(merged, donor_bc)

    # Force block_count=1, hash_layer_count=0
    force_override(merged, ".block_count", 1)       # P5
    force_override(merged, ".hash_layer_count", 0)  # P8

    arch = "deepseek4"
    print(f"Target arch: {arch}")
    print(f"Merged keys: {len(merged)}")
    print(f"MTP tensors: {len(reader.tensors)}")

    writer = GGUFWriter(output_path, arch)
    write_metadata(writer, merged)

    print(f"\nAdding tensors...")
    for name in DONOR_TENSORS:
        if name in donor_tensors:
            t = donor_tensors[name]
            writer.add_tensor(name, t.data, raw_dtype=t.tensor_type)
            print(f"  [+donor] {name}")

    for i, rt in enumerate(reader.tensors):
        new_name = rt.name
        if rt.name.startswith("mtp.0."):
            new_name = "blk.0." + rt.name[len("mtp.0."):]
        if new_name != rt.name:
            print(f"  [{i}] {rt.name} -> {new_name}")
        writer.add_tensor(new_name, rt.data, raw_dtype=rt.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(1)

    inp = sys.argv[1]
    donor = None
    out = "mtp-converted-deepseek4.gguf"
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--donor" and i + 1 < len(args):
            donor = args[i + 1]; i += 2
        elif not args[i].startswith("--"):
            out = args[i]; i += 1
        else:
            i += 1

    if not os.path.exists(inp):
        print(f"Error: MTP not found: {inp}"); sys.exit(1)
    if not donor:
        print("Error: --donor required"); sys.exit(1)
    if not os.path.exists(donor):
        print(f"Error: donor not found: {donor}"); sys.exit(1)

    convert(inp, donor, out)
