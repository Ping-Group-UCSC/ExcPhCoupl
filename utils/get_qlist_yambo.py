#!/cm/shared/apps/python/3.8.6/bin/python
import sys
import re
import math

def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results



print("\n\n * * * Get q-points from Ypp output * * *")

qstart=search_string_in_file('r_gkkp_gkkp_db','Q-points list in Yambo')
qend  =search_string_in_file('r_gkkp_gkkp_db','[09] Timing Overview')

nq_points=qend[0][0]-qstart[0][0]-4

print("Number of q-points : ",nq_points)

ypp_log=open('r_gkkp_gkkp_db','r')
lines=ypp_log.readlines()

qfile=open('qpoints_yambo','w')
qfile.write(str(nq_points)+"\n")

for il in range(qstart[0][0]+2,qend[0][0]-2):
    qpoint=[round(float(item),6) for item in lines[il].split()]
#    qpoint=[float(item) for item in lines[il].split()]
    qfile.write(str(qpoint[0])+"  "+str(qpoint[1])+"  "+str(qpoint[2])+"  1\n")

qfile.close()
