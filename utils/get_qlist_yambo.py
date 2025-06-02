#!/cm/shared/apps/python/3.8.6/bin/python
import sys
import re
import math
import os

# ──────────────────────────────────────────────────────────────────────────────
# 1) Change this line to point at your “exciton” folder:
# ──────────────────────────────────────────────────────────────────────────────
YAMBO_DATA_DIR = "/Users/keyneshdongol/Downloads/excph-devel(1)/yambo-qe-nk662-exp-lat/QPT6/dvscf/bn.save"


# ──────────────────────────────────────────────────────────────────────────────

def search_string_in_file(file_name, string_to_search):
    """Search for string_to_search in file_name, return list of (lineno, line)."""
    line_number = 0
    list_of_results = []
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                list_of_results.append((line_number, line.rstrip()))
    return list_of_results


print("\n\n * * * Get q-points from Ypp output * * *")

# Build the full path to r_gkkp_gkkp_db inside the “exciton” folder:
yambo_db_path = os.path.join(YAMBO_DATA_DIR, "r_gkkp_gkkp_db")

# Now search inside that file:
qstart = search_string_in_file(yambo_db_path, 'Q-points list in Yambo')
qend   = search_string_in_file(yambo_db_path, '[09] Timing Overview')

# If the markers are found, compute # of q‐points:
if not qstart or not qend:
    print("ERROR: Could not find the markers in", yambo_db_path)
    sys.exit(1)

nq_points = qend[0][0] - qstart[0][0] - 4
print("Number of q-points:", nq_points)

# Read all lines so we can slice out the q‐point block:
with open(yambo_db_path, 'r') as ypp_log:
    lines = ypp_log.readlines()

# Open (or create) the output file “qpoints_yambo” inside the same directory:
qfile_path = os.path.join(YAMBO_DATA_DIR, "qpoints_yambo")
with open(qfile_path, 'w') as qfile:
    qfile.write(str(nq_points) + "\n")

    # Write each q‐point (x, y, z, 1) as in the original script:
    for il in range(qstart[0][0] + 2, qend[0][0] - 2):
        qpoint = [round(float(item), 6) for item in lines[il].split()]
        qfile.write(f"{qpoint[0]}  {qpoint[1]}  {qpoint[2]}  1\n")

print("Wrote qpoints_yambo →", qfile_path)
