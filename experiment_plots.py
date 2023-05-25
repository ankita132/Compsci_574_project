import matplotlib.pyplot as plt

original_accuracy = [0.317600, 0.359370, 0.389451, 0.427869, 0.446753, 0.491245, 0.509856, 0.516781, 0.579876, 0.587643]

new_accuracy = [0.046436, 0.048390, 0.049532, 0.054936, 0.057832, 0.059952, 0.066541, 0.071123, 0.073189, 0.073534]

epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

plt.plot(epochs, original_accuracy, label='original model')
plt.plot(epochs, new_accuracy, label='proposed model')

plt.title("Mean IOU on test data on both models (10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("mean IOU")

plt.legend()
plt.savefig('plot2.png')

plt.show()
