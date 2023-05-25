import matplotlib.pyplot as plt

original_accuracy = [0.363590, 0.364200, 0.361790, 0.360807, 0.360139]

# new_accuracy = [0.046436, 0.048390, 0.049532, 0.054936, 0.057832, 0.059952, 0.066541, 0.071123, 0.073189, 0.073534]

dropout = [0, 0.1, 0.3, 0.5, 0.7]

plt.plot(dropout, original_accuracy)
# plt.plot(epochs, new_accuracy, label='proposed model')

plt.title("Test Accuracy vs Dropout")
plt.xlabel("Dropout")
plt.ylabel("Accuracy")

# plt.legend()
plt.savefig('plot4.png')

plt.show()
