"""
rotate_transform.py
-------------------
Read 3D vector data (Euler angles, angular velocity, or acceleration) from a
text DAT file, rotate every vector by three user-supplied angles (Rx, Ry, Rz in
degrees), and write the result to a new text file in the same format.

Input file format
-----------------
The first column is a label / time column and is copied unchanged.
Every subsequent group of three columns is treated as a 3-D vector and is
rotated by the combined rotation matrix  R = Rz @ Ry @ Rx.

Example header:
    time,roll,pitch,yaw,wx,wy,wz,ax,ay,az

Usage
-----
    python rotate_transform.py                          (uses defaults below)
    python rotate_transform.py input.dat output.dat     (filenames as args)
"""

import math


# ---------------------------------------------------------------------------
# User configuration – edit these values before running the script
# ---------------------------------------------------------------------------

INPUT_FILE  = "kinematics_log_body_001.dat"   # input file
OUTPUT_FILE = "kinematics_log_body_001_rotated.dat"   # output file

# Rotation angles in degrees (Rx then Ry then Rz)
RX_DEG = 0.0
RY_DEG = 0.0
RZ_DEG = 0.0


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float):
    """Return the combined 3x3 rotation matrix R = Rz @ Ry @ Rx."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    cos_z, sin_z = math.cos(rz), math.sin(rz)

    # Rx
    Rx = [
        [1,      0,       0],
        [0,  cos_x, -sin_x],
        [0,  sin_x,  cos_x],
    ]
    # Ry
    Ry = [
        [ cos_y, 0, sin_y],
        [     0, 1,     0],
        [-sin_y, 0, cos_y],
    ]
    # Rz
    Rz = [
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [    0,      0, 1],
    ]

    return mat_mul(mat_mul(Rz, Ry), Rx)


def mat_mul(A, B):
    """Multiply two 3x3 matrices."""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result


def apply_rotation(R, vec):
    """Apply rotation matrix R to a 3-element vector."""
    return [
        R[0][0] * vec[0] + R[0][1] * vec[1] + R[0][2] * vec[2],
        R[1][0] * vec[0] + R[1][1] * vec[1] + R[1][2] * vec[2],
        R[2][0] * vec[0] + R[2][1] * vec[1] + R[2][2] * vec[2],
    ]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_text_table(filepath: str):
    """Return (header_row, data_rows, trailing_comma).

    Parses comma-separated text with optional whitespace padding and a trailing
    comma on data lines (common in .dat exports).
    """
    with open(filepath, "r", newline="") as fh:
        raw_lines = [line.rstrip("\n") for line in fh]
    lines = [line for line in raw_lines if line.strip()]
    if not lines:
        raise ValueError(f"Input file '{filepath}' is empty.")

    def split_row(line: str):
        parts = [part.strip() for part in line.split(",")]
        had_trailing = False
        if parts and parts[-1] == "":
            had_trailing = True
            parts = parts[:-1]
        return parts, had_trailing

    header, _ = split_row(lines[0])
    data_rows = []
    trailing_comma = False
    for line in lines[1:]:
        row, had_trailing = split_row(line)
        trailing_comma = trailing_comma or had_trailing
        data_rows.append(row)
    return header, data_rows, trailing_comma


def write_text_table(filepath: str, header: list, data_rows: list, trailing_comma: bool):
    """Write header + data_rows to a comma-separated text file."""
    with open(filepath, "w", newline="") as fh:
        fh.write(", ".join(header) + "\n")
        for row in data_rows:
            line = ", ".join(str(value) for value in row)
            if trailing_comma:
                line += ","
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def transform_rows(data_rows: list, R):
    """
    Rotate every 3D vector in each data row.

    Column layout expected:
        col 0          : time / label  (copied unchanged)
        cols 1-3       : first 3-D vector  (rotated)
        cols 4-6       : second 3-D vector (rotated), if present
        cols 7-9       : third 3-D vector  (rotated), if present
        ...

    Any trailing columns that do not form a complete triple are copied as-is.
    """
    out = []
    for row in data_rows:
        new_row = [row[0]]          # keep label / time column unchanged
        idx = 1
        while idx + 3 <= len(row):  # need at least 3 values starting at idx
            try:
                vec = [float(row[idx]), float(row[idx + 1]), float(row[idx + 2])]
            except ValueError:
                # Non-numeric values: copy verbatim
                new_row.extend(row[idx:idx + 3])
                idx += 3
                continue
            rotated = apply_rotation(R, vec)
            new_row.extend(rotated)
            idx += 3
        # copy any leftover columns unchanged
        new_row.extend(row[idx:])
        out.append(new_row)
    return out


def _detect_column_formats(data_rows: list):
    """Infer numeric formatting (scientific + decimals) from input strings."""
    formats = {}
    max_cols = max((len(r) for r in data_rows), default=0)
    for col in range(max_cols):
        for row in data_rows:
            if col >= len(row):
                continue
            token = row[col]
            try:
                float(token)
            except ValueError:
                continue
            if "e" in token.lower():
                mantissa = token.split("e")[0]
                decimals = mantissa.split(".")[1] if "." in mantissa else ""
                formats[col] = ("scientific", len(decimals))
            else:
                decimals = token.split(".")[1] if "." in token else ""
                formats[col] = ("fixed", len(decimals))
            break
    return formats


def format_rows(data_rows: list, column_formats: dict):
    """Format float values using the same precision/style as input data."""
    formatted = []
    for row in data_rows:
        new_row = []
        for col, val in enumerate(row):
            if isinstance(val, float):
                fmt = column_formats.get(col)
                if fmt is None:
                    new_row.append(repr(val))
                else:
                    style, decimals = fmt
                    if style == "scientific":
                        new_row.append(f"{val:.{decimals}E}")
                    else:
                        new_row.append(f"{val:.{decimals}f}")
            else:
                new_row.append(val)
        formatted.append(new_row)
    return formatted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== RotateTransform ===")
    print("Rotates 3-D vector data (Euler angles, angular velocity, acceleration)")
    print("by user-specified angles around the X, Y, and Z axes.\n")

    # --- filenames ---
    input_file = INPUT_FILE
    output_file = OUTPUT_FILE

    # --- rotation angles ---
    rx = RX_DEG
    ry = RY_DEG
    rz = RZ_DEG

    # --- read ---
    print(f"\nReading '{input_file}' ...")
    header, data_rows, trailing_comma = read_text_table(input_file)

    if not data_rows:
        print("Warning: no data rows found in the input file.")

    # --- transform ---
    R = rotation_matrix(rx, ry, rz)
    transformed = transform_rows(data_rows, R)
    column_formats = _detect_column_formats(data_rows)
    formatted = format_rows(transformed, column_formats)

    # --- write ---
    write_text_table(output_file, header, formatted, trailing_comma)
    print(f"Transformed data written to '{output_file}'.")
    print(f"  Rows processed : {len(data_rows)}")
    print(f"  Rotation applied: Rx={rx}°  Ry={ry}°  Rz={rz}°")


if __name__ == "__main__":
    main()
