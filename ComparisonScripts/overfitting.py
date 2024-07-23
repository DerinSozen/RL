import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(1, 10, 100)
ys1 = -(xs**2) + 10
ys2 = -0.1*(xs**2) + 10

plt.plot(xs, ys1,label="Training Loss")
plt.plot(xs,ys2,label="Validation Loss")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Overfitting Example, Worst validation, Best training")
plt.savefig("BestTrainingWorstValidation.png")