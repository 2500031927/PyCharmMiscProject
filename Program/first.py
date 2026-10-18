import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("All libraries imported successfully!")

arr = np.array([10, 20, 30, 40])
print("NumPy Array:", arr)

data = pd.DataFrame({
    "Name": ["Ali", "Sara", "John"],
    "Marks": [85, 90, 78]
})

print("\nDataFrame:")
print(data)

plt.plot([1, 2, 3, 4], [10, 20, 15, 25])
plt.title("Sample Graph")
plt.show()