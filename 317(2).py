#wiki生成 电场推介电常数分布
#修改：学习率、神经元数量、波长单位

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


# 1. Load and prepare data
def load_data(file_path):
    data = np.loadtxt(file_path)
    x = data[:, 0:1]  # x coordinates
    y = data[:, 1:2]  # y coordinates
    E = data[:, 2:3]  # Electric field values
    return x, y, E


file_path = "D:/Anaconda/PythonProject2/317(e=3 wavelength=2100)%.txt"
x_data, y_data, E_data = load_data(file_path)
xy_data = np.hstack((x_data, y_data))

# 2. Define geometry
x_min, x_max = np.min(x_data), np.max(x_data)
y_min, y_max = np.min(y_data), np.max(y_data)
geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])


# 3. Define PDE relating electric field to permittivity
def pde(x, y):
    E, epsilon = y[:, 0:1], y[:, 1:2]
    E_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    E_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    k0 = 2 * np.pi / 2.1  # Wavenumber for 2100 nm wavelength
    return E_xx + E_yy + (k0 ** 2) * epsilon * E


# 4. Define boundary conditions using measurement data
observe_E = dde.icbc.PointSetBC(xy_data, E_data, component=0)

# 5. Create PDE problem
data = dde.data.PDE(
    geom,
    pde,
    [observe_E],
    num_domain=2000,
    num_boundary=100,
    anchors=xy_data
)

# 6. Create neural network
net = dde.nn.PFNN(
    [2, 64, 64, 64, 64, 2],
    #[2, [40, 40], [40, 40], [40, 40], 2],
    "tanh",
    "Glorot uniform"
)

# 7. Create and train model
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1, 100])
losshistory, train_state = model.train(iterations=10)

# 8. Visualize results
# Create meshgrid for prediction
x_range = np.linspace(x_min, x_max, 100)
y_range = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_range, y_range)
X_flat = X.flatten()[:, None]
Y_flat = Y.flatten()[:, None]
points = np.hstack((X_flat, Y_flat))

# Predict on meshgrid
predictions = model.predict(points)
E_pred = predictions[:, 0].reshape(X.shape)
epsilon_pred = predictions[:, 1].reshape(X.shape)

# Plot permittivity distribution
plt.figure(figsize=(10, 8))
plt.pcolormesh(X, Y, epsilon_pred, shading='auto', cmap='viridis')
plt.colorbar(label='Permittivity (ε)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Predicted Permittivity Distribution')
plt.savefig('permittivity_distribution.png')
plt.show()

# Save permittivity data
np.savetxt('permittivity_data.txt',
           np.column_stack((X_flat, Y_flat, predictions[:, 1])),
           header='X Y Permittivity',
           delimiter='\t')