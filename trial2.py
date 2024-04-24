import numpy as np

# Function to calculate angle between two 3D vectors (-pi to pi)
def angle_between_vectors_3d(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_theta = dot_product / (norm_vec1 * norm_vec2)

    # Clip the cosine value to [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calculate the angle using arccosine
    angle_rad = np.arccos(cos_theta)

    # Calculate the sign of the angle using the cross product
    cross_product = np.cross(vec1, vec2)
    if np.dot(cross_product, cross_product) < 1e-15:
        sign = 0
    else:
        sign = np.sign(np.dot(cross_product, np.array([0, 0, 1])))

    # Adjust the angle sign based on the cross product
    angle_rad *= sign

    return angle_rad

# Test cases
vec1 = np.array([1, 0, 0])  # X-axis vector
vec2 = np.array([0, 1, 0])  # Y-axis vector
print("Angle between X-axis and Y-axis vectors:", angle_between_vectors_3d(vec1, vec2))

vec1 = np.array([1, 0, 0])  # X-axis vector
vec2 = np.array([1, 1, 0])  # 45-degree vector in XY plane
print("Angle between X-axis and 45-degree vector in XY plane:", angle_between_vectors_3d(vec1, vec2))

vec1 = np.array([1, 0, 0])  # X-axis vector
vec2 = np.array([-1, 0, 0])  # Opposite X-axis vector
print("Angle between opposite X-axis vectors:", angle_between_vectors_3d(vec1, vec2))

vec1 = np.array([1, 0, 0])  # X-axis vector
vec2 = np.array([0, -1, 0])  # Y-axis vector in opposite direction
print("Angle between X-axis and opposite Y-axis vector:", angle_between_vectors_3d(vec1, vec2))

vec1 = np.array([1, 0, 0])  # X-axis vector
vec2 = np.array([0, 0, 1])  # Z-axis vector
print("Angle between X-axis and Z-axis vector:", angle_between_vectors_3d(vec1, vec2))

vec1 = np.array([1, 1, 1])  # Vector in XYZ diagonal direction
vec2 = np.array([-1, -1, -1])  # Opposite vector in opposite XYZ diagonal direction
print("Angle between opposite vectors in XYZ diagonal directions:", angle_between_vectors_3d(vec1, vec2))
