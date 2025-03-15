import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the dataset
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# lifesat is Data Frame

X = lifesat[["GDP per capita (USD)"]].values # second row
y = lifesat[["Life satisfaction"]].values # third row
# print(X)
# print(y)

# Visualize the dataset
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
print(model.predict(X_new)) # outputs [[6.30165767]]

# Replacing the Linear Regression model with k-Nearest Neighbors
# (in this example, k = 3) regression:
from sklearn.neighbors import KNeighborsRegressor
new_model = KNeighborsRegressor(n_neighbors=3)

# Train the model
new_model.fit(X, y)
# Make a prediction for Cyprus
print(new_model.predict(X_new)) # outputs [[6.33333333]]




# Save figures - (can pass!!)

from pathlib import Path
# Where to save the figures
IMAGES_PATH = Path() / "images" / "fundamentals"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
