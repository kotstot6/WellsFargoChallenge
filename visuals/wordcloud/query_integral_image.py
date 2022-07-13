import array
import numpy as np
import matplotlib.pyplot as plt

def distance(ij, pos):
    i, j = ij
    return np.linalg.norm(pos - np.array([i, j]))

def query_integral_image(integral_image, size_x, size_y, random_state, pos, eps1, eps2):
    x = integral_image.shape[0]
    y = integral_image.shape[1]
    i, j = 0, 0
    hits = []

    # count how many possible locations
    for i in range(x - size_x):
        for j in range(y - size_y):
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            if not area:
                d = distance((i, j), pos)
                if d <= eps1:
                    hits.append((i,j,d))
    if not hits:
        # no room left
        return None

    best_hits = [(i, j) for i, j, d in hits if d <= eps2 * eps1]

    if best_hits:
        return best_hits[np.random.choice(len(best_hits))]
    else:
        i, j, _ = min(hits, key=lambda x: x[2])
        return i, j

    """
    # pick a location at random
    goal = random_state.randint(0, hits)
    hits = 0
    for i in range(x - size_x):
        for j in range(y - size_y):
            area = integral_image[i, j] + integral_image[i + size_x, j + size_y]
            area -= integral_image[i + size_x, j] + integral_image[i, j + size_y]
            if not area:
                hits += 1
                if hits == goal:
                    print(i, j)
                    return i, j
    """
