from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Save Figures
# Where to save the figures
IMAGES_PATH = Path() / "images" / "fundamentals"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# variables of program

dataPath = Path() / "datasets" / "lifesat"
dataPath.mkdir(parents=True, exist_ok=True)

data_root = "https://github.com/ageron/data/raw/main/"
filenames = ["oecd_bli.csv", "gdp_per_capita.csv"]

# Download Dataset from path
def downloadData():
    for filename in filenames:
        if not (dataPath / filename).is_file():
            print("Downloading", filename)
            url = data_root + "lifesat/" + filename
            urllib.request.urlretrieve(url, dataPath / filename)


# downloadData()

gdp_per_capita = pd.read_csv(dataPath / "gdp_per_capita.csv")
oecd_bli = pd.read_csv(dataPath / "oecd_bli.csv")

# Preprocess the GDP per capita dataset to keep only the year 2020:
gdp_year = 2020
gdppc_col = "GDP per capita (USD)"
gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == gdp_year]  # filter the year = 2020
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)  # remove column has name "Code" and "Year"
gdp_per_capita.columns = ["Country", gdppc_col]  # two new columns
gdp_per_capita.set_index("Country", inplace=True)
print(gdp_per_capita.head())

# Preprocess the OECD BLI dataset to keep only the Life satisfaction column:
lifesat_col = "Life satisfaction"
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
print(oecd_bli.head())

# Now let's merge the life satisfaction dataset and the GDP per capita dataset,
# keeping only the GDP per capita and Life satisfaction columns:

full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
full_country_stats.sort_values(by=gdppc_col, inplace=True)
full_country_stats = full_country_stats[[gdppc_col, lifesat_col]]

print(full_country_stats.head())

# To illustrate the risk of overfitting, I use only part of the dataset in most figures (all countries with
# a GDP per capita between min_gdp and max_gdp).
# Later in the chapter I reveal the missing countries, and show that they don't follow the same linear trend at all.

min_gdp, max_gdp = 23500, 62500
country_stats = full_country_stats[(full_country_stats[gdppc_col] >= min_gdp) &
                                   (full_country_stats[gdppc_col] <= max_gdp)]

print(country_stats)

# save csv
country_stats.to_csv(dataPath / 'lifesat.csv')
full_country_stats.to_csv(dataPath / 'lifesat_full')

country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)

min_life_sat, max_life_sat = 4, 9

position_text = {
    "Turkey": (29_500, 4.2),
    "Hungary": (28_000, 6.9),
    "France": (40_000, 5),
    "New Zealand": (28_000, 8.2),
    "Australia": (50_000, 5.5),
    "United States": (59_000, 5.3),
    "Denmark": (46_000, 8.5)
}

for country, pos_text in position_text.items():
    pos_data_x = country_stats[gdppc_col].loc[country]
    pos_data_y = country_stats[lifesat_col].loc[country]
    country = "U.S" if country == "United States" else country

    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5, shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "r")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])
save_fig('money_happy_scatterplot')
plt.show()

highlighted_countries = country_stats.loc[list(position_text.keys())]
highlighted_countries[[gdppc_col, lifesat_col]].sort_values(by=gdppc_col)

print(highlighted_countries)


country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)
X = np.linspace(min_gdp, max_gdp, 1000)

w1, w2 = 4.2, 0
plt.plot(X, w1 + w2 * 1e-5 * X, "r")
plt.text(40_000, 4.9, fr"$\theta_0 = {w1}$", color="r")
plt.text(40_000, 4.4, fr"$\theta_1 = {w2}$", color="r")

w1, w2 = 10, -9
plt.plot(X, w1 + w2 * 1e-5 * X, "g")
plt.text(26_000, 8.5, fr"$\theta_0 = {w1}$", color="g")
plt.text(26_000, 8.0, fr"$\theta_1 = {w2} \times 10^{{-5}}$", color="g")

w1, w2 = 3, 8
plt.plot(X, w1 + w2 * 1e-5 * X, "b")
plt.text(48_000, 8.5, fr"$\theta_0 = {w1}$", color="b")
plt.text(48_000, 8.0, fr"$\theta_1 = {w2} \times 10^{{-5}}$", color="b")

plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])
save_fig("tweacking_model_params_plot")
plt.show()

from sklearn import linear_model

X_sample = country_stats[[gdppc_col]].values
Y_sample = country_stats[[lifesat_col]].values

lin1 = linear_model.LinearRegression()
lin1.fit(X_sample, Y_sample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
print(f"θ0={t0:.2f}, θ1={t1:.2e}")

country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)
X = np.linspace(min_gdp, max_gdp, 1000)
plt.plot(X, t0 + t1 * X, "b")
plt.text(max_gdp - 20_000, min_life_sat + 1.9, fr"$\theta_0 = {t0:.2f}$", color="b")
plt.text(max_gdp - 20_000, min_life_sat + 1.3, fr"$\theta_1 = {t1 * 1e5:.2f} \times 10^{{-5}}$", color="b")
plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])
save_fig("best_fit_model_plot")
plt.show()

# Predict for Cyprus gdp per capital
cyprus_gdp_per_capita = gdp_per_capita[gdppc_col].loc["Cyprus"]
print('Cyprus GDP per Capital: ', cyprus_gdp_per_capita)

cyprus_predicted_life_satisfaction = lin1.predict([[cyprus_gdp_per_capita]])[0][0]
print('Predict for Cyprus GDP per Capital', cyprus_predicted_life_satisfaction)

country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)
X = np.linspace(min_gdp, max_gdp, 1000)
plt.plot(X, t0 + t1 * X, "b")
plt.text(max_gdp - 20_000, min_life_sat + 1.9, fr"$\theta_0 = {t0:.2f}$", color="b")
plt.text(max_gdp - 20_000, min_life_sat + 1.3, fr"$\theta_1 = {t1 * 1e5:.2f} \times 10^{{-5}}$", color="b")

plt.plot([cyprus_gdp_per_capita, cyprus_gdp_per_capita],
         [min_life_sat, cyprus_predicted_life_satisfaction], 'r--')
plt.text(cyprus_gdp_per_capita + 1000, 5.0, fr"Prediction = {cyprus_predicted_life_satisfaction:.2f}", color="r")
plt.plot(cyprus_gdp_per_capita, cyprus_predicted_life_satisfaction, "ro")
plt.axis([min_gdp, max_gdp, min_life_sat, max_life_sat])
plt.show()

# find missing dataset
missing_data = full_country_stats[(full_country_stats[gdppc_col] < min_gdp) | (full_country_stats[gdppc_col] > max_gdp)]
print(missing_data)

position_text_misssing_countries = {
    "South Africa": (20_000, 4.2),
    "Colombia": (6_000, 8.2),
    "Brazil": (18_000, 7.8),
    "Mexico": (24_000, 7.4),
    "Chile": (30_000, 7.0),
    "Norway": (51_000, 6.2),
    "Switzerland": (62_000, 5.7),
    "Ireland": (81_000, 5.2),
    "Luxembourg": (92_000, 4.7),
}

full_country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)
for country, pos_text in position_text_misssing_countries.items():
    pos_data_x, pos_data_y = missing_data.loc[country]
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5, shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, 'rs')
X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1 * X, "b:")

lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[full_country_stats[gdppc_col]]
Yfull = np.c_[full_country_stats[lifesat_col]]
lin_reg_full.fit(Xfull, Yfull)

t0_full, t1_full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0_full + t1_full * X, 'k')
plt.axis([0, 115_000, min_life_sat, max_life_sat])
save_fig('representative_training_data_scatterplot')
plt.show()


# Pipeline
from sklearn import preprocessing
from sklearn import pipeline
full_country_stats.plot(kind='scatter', figsize=(5, 3), grid=True, x=gdppc_col, y=lifesat_col)

poly = preprocessing.PolynomialFeatures(degree=10, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([
    ('poly', poly),
    ('scal', scaler),
    ('lin', lin_reg2)
])

pipeline_reg.fit(Xfull, Yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
plt.axis([0, 115_000, min_life_sat, max_life_sat])
save_fig('overfitting_model_plot')
plt.show()

w_countries = [c for c in full_country_stats.index if "W" in c.upper()]
print(full_country_stats.loc[w_countries][lifesat_col])

all_w_countries = [c for c in gdp_per_capita.index if "W" in c.upper()]
print(gdp_per_capita.loc[all_w_countries].sort_values(by=gdppc_col))

# Using Linear Model Ridge
country_stats.plot(kind='scatter', x=gdppc_col, y=lifesat_col, figsize=(8, 3))
missing_data.plot(kind='scatter', x=gdppc_col, y=lifesat_col,
                  marker="s", color="r", grid=True, ax=plt.gca())

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1*X, "b:", label="Linear model on partial dataset")
plt.plot(X, t0_full + t1_full * X, "k-", label="Linear model on all dataset")

ridge = linear_model.Ridge(alpha=10**9.5)
X_sample = country_stats[[gdppc_col]]
y_sample = country_stats[[lifesat_col]]
ridge.fit(X_sample, y_sample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0][0]
plt.plot(X, t0ridge + t1ridge * X, "b--", label="Regularized linear model on partial dataset")
plt.legend(loc="lower right")

plt.axis([0, 115_000, min_life_sat, max_life_sat])
save_fig('ridge_model_plot')
plt.show()