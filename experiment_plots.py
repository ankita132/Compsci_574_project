import matplotlib.pyplot as plt

original_accuracy = [0.360801, 0.363590, 0.368985, 0.380376, 0.369139]

# new_accuracy = [0.046436, 0.048390, 0.049532, 0.054936, 0.057832, 0.059952, 0.066541, 0.071123, 0.073189, 0.073534]

alpha = [0.5, 0.2, 0.1, 0.05, 0.01]

plt.plot(alpha, original_accuracy)
# plt.plot(epochs, new_accuracy, label='proposed model')

plt.title("Test Accuracy vs Alpha")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")

# plt.legend()
plt.savefig('plot3.png')

plt.show()
