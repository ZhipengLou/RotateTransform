"""
rotate_transform.py
-------------------
Read 3D vector data (Euler angles, angular velocity, or acceleration) from a
CSV/DAT file, rotate every vector by three user-supplied angles (Rx, Ry, Rz in
degrees), and write the result to a new CSV file in the same format.

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
    python rotate_transform.py input.csv output.csv     (filenames as args)
"""

import sys
import csv
import math


# ---------------------------------------------------------------------------
# User configuration – edit these values before running the script
# ---------------------------------------------------------------------------

INPUT_FILE  = "kinematics_log_body_001.dat"   # example input file
OUTPUT_FILE = "output.csv"                     # default output file


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

def read_csv(filepath: str):
    """Return (header_row, data_rows) where each row is a list of strings.

    Handles files where values are padded with whitespace and rows end with a
    trailing comma (common in .dat exports).
    """
    with open(filepath, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input file '{filepath}' is empty.")
    # Strip leading/trailing whitespace from every cell and remove trailing
    # empty fields produced by a trailing comma.
    cleaned = []
    for row in rows:
        stripped = [cell.strip() for cell in row]
        last_idx = len(stripped) - 1
        while last_idx >= 0 and stripped[last_idx] == "":
            last_idx -= 1
        cleaned.append(stripped[:last_idx + 1])
    return cleaned[0], cleaned[1:]


def write_csv(filepath: str, header: list, data_rows: list):
    """Write header + data_rows to a CSV file."""
    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(data_rows)


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


def format_rows(data_rows: list, decimal_places: int = 6):
    """Round float values to a fixed number of decimal places."""
    formatted = []
    for row in data_rows:
        new_row = []
        for val in row:
            if isinstance(val, float):
                new_row.append(round(val, decimal_places))
            else:
                new_row.append(val)
        formatted.append(new_row)
    return formatted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_filename(prompt: str, argv_index: int, default: str = "") -> str:
    """Return a filename from argv, interactive input, or the supplied default."""
    if len(sys.argv) > argv_index:
        return sys.argv[argv_index]
    display = f"{prompt}: [{default}]: " if default else f"{prompt}: "
    value = input(display).strip()
    return value if value else default


def get_angle(axis: str) -> float:
    while True:
        raw = input(f"  Rotation angle around {axis}-axis (degrees): ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a valid number.")


def main():
    print("=== RotateTransform ===")
    print("Rotates 3-D vector data (Euler angles, angular velocity, acceleration)")
    print("by user-specified angles around the X, Y, and Z axes.\n")

    # --- filenames ---
    input_file = get_filename("Input file path ", 1, INPUT_FILE)
    output_file = get_filename("Output file path", 2, OUTPUT_FILE)

    # --- rotation angles ---
    print("\nEnter the rotation angles (applied as Rx first, then Ry, then Rz):")
    rx = get_angle("X")
    ry = get_angle("Y")
    rz = get_angle("Z")

    # --- read ---
    print(f"\nReading '{input_file}' ...")
    header, data_rows = read_csv(input_file)

    if not data_rows:
        print("Warning: no data rows found in the input file.")

    # --- transform ---
    R = rotation_matrix(rx, ry, rz)
    transformed = transform_rows(data_rows, R)
    formatted = format_rows(transformed)

    # --- write ---
    write_csv(output_file, header, formatted)
    print(f"Transformed data written to '{output_file}'.")
    print(f"  Rows processed : {len(data_rows)}")
    print(f"  Rotation applied: Rx={rx}°  Ry={ry}°  Rz={rz}°")


if __name__ == "__main__":
    main()
