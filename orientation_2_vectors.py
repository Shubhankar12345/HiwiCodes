import numpy as np
 
def unit_vector(v1):
    "Returns the unit vector for a given vector"
    return v1/np.linalg.norm(v1)

def diff_orientation(v1,v2):
    "Returns the differnce in orientation between two vectors"
    vec1 = unit_vector(v1)
    vec2 = unit_vector(v2)
    cosine_vecs = np.dot(vec1,vec2)
    sine_vecs = np.cross(vec1,vec2)
    if(cosine_vecs == 0):
        return np.arctan2(np.min(np.sign(sine_vecs)),cosine_vecs)
    else:          
        return np.arctan2(np.linalg.norm(sine_vecs),cosine_vecs)

# Test cases
test_cases = [(np.array([1, 0, 0]),np.array([0, 1, 0])), # X-axis vector and Y-axis vector
               (np.array([1, 0, 0]),np.array([1, 1, 0])), # X-axis vector and 45-degree vector in XY plane
               (np.array([1, 0, 0]),np.array([-1, 0, 0])), # X-axis vector and opposite X axis vector
               (np.array([1, 0, 0]),np.array([0, -1, 0])), # X-axis vector and Y-axis vector in opposite direction
               (np.array([1, 0, 0]),np.array([0, 0, 1])), # X-axis vector and Z-axis vector in opposite direction
               (np.array([1, 1, 1]),np.array([-1, -1, -1])), # Vector in XYZ diagonal direction and vector in XYZ opposite diagonal direction
               (np.array([1, -1, 0]),np.array([-1, -1, 0])), # vectors in 3rd and 4th quadrant
               (np.array([5.45,1.12]),np.array([-3.86,4.32]))] # Random vectors with 120 degree angle between them

for index, (a,b) in enumerate(test_cases):
    print(f"Angle between the vectors = {diff_orientation(a,b)}")
