# Zomato Restaurant Segmentation using K-Means Clustering

An unsupervised machine learning project that segments restaurants into natural business-relevant groups — without any predefined labels — using price, cuisine, popularity, and customer engagement data.

## Problem Statement

Zomato lists thousands of restaurants with different prices, cuisines, ratings, and popularity levels. Analyzing and managing them individually doesn't scale, and there are no predefined categories for grouping similar restaurants. This project uses **K-Means clustering**, an unsupervised learning algorithm, to discover natural restaurant groupings that can inform pricing, marketing, and customer-targeting strategy.

## Datasets

| Dataset | Rows | Description |
|---|---|---|
| `Zomato Restaurant names and Metadata.csv` | 105 | Restaurant name, cost for two, cuisines, curated Zomato collections, timings |
| `Zomato Restaurant reviews.csv` | ~10,000 | Individual customer reviews across 100 of those restaurants — reviewer, review text, star rating, reviewer metadata, timestamp, picture count |

## Methodology

1. **Cleaning** — converted `Cost` from comma-separated text to numeric, parsed `Cuisines`/`Collections` into counts, dropped 36 blank review rows, converted `Rating` to numeric (one invalid `'Like'` value became `NaN`), parsed reviewer metadata out of a free-text field with regex.
2. **Feature engineering** — aggregated ~10,000 individual reviews into one row per restaurant (`avg_rating`, `rating_std`, `avg_pictures`, `avg_reviewer_followers`) *before* merging, to avoid row-duplication bias.
3. **Merging** — left-joined metadata (105 restaurants) with aggregated review features (100 restaurants) on restaurant name, keeping all 105 restaurants.
4. **Missing data** — added a `has_reviews` flag for the 5 restaurants with no reviews, then imputed their review-derived features with the median (not zero) of the 100 reviewed restaurants.
5. **EDA** — 16 univariate, bivariate, and multivariate charts to understand distributions and relationships before selecting clustering features.
6. **Feature selection** — chose `Cost`, `num_cuisines`, `num_collections`, `avg_rating`, `rating_std`, `avg_pictures` for clustering; deliberately excluded `review_count` (near-zero variance across restaurants) and `avg_reviewer_followers` (describes reviewers, not restaurants).
7. **Scaling** — applied `StandardScaler` so no single feature (e.g. `Cost`, which is in the thousands) dominates the Euclidean distance calculation.
8. **Choosing k** — evaluated k=2 through k=10 with the Elbow Method and Silhouette Score, and compared full cluster profiles for k=3, k=7, and k=8.
9. **Model selection: k=3** — the second-best silhouette score (0.272, after k=2's 0.324, which only separated restaurants by price), aligned with the elbow's bend, and produced three robust, business-meaningful clusters (unlike k=8, which produced a fragile 3-restaurant cluster).
10. **Validation** — cross-checked K-Means against Agglomerative (Hierarchical) Clustering; Adjusted Rand Index of 0.44 indicated moderate agreement, supporting the 3-cluster structure independent of the algorithm used.

## Results — Three Clusters

| Cluster | Size | Avg. Cost | Avg. Rating | Rating Consistency | Profile |
|---|---|---|---|---|---|
| **Budget Everyday** | 60 (57.1%) | ₹553 | 3.38 | Most variable | Largest segment; lowest cost and rating; most polarizing |
| **Mid-Premium / Photogenic** | 30 (28.6%) | ₹1,197 | 3.62 | Moderate | Highest photo engagement (avg. 1.52 pictures/review) |
| **Premium & Curated** | 15 (14.3%) | ₹1,423 | 4.39 | Most consistent | Highest rating and most Zomato collections; lowest rating variance |

**Final silhouette score:** 0.272 (moderate — clusters are identifiable but have some overlap, reported honestly rather than overstated).

**Scope caveat:** these clusters describe this specific dataset (105 restaurants, apparently a single city/region, reviews collected up to a specific point in time). They should not be assumed to generalize to Zomato's full or current restaurant base without re-running the analysis on fresh data. All relationships reported (e.g. cost vs. rating, collections vs. rating) are correlational, not causal.

## Repository Structure

```
zomato-restaurant-clustering/
├── data/
│   ├── Zomato Restaurant names and Metadata.csv
│   └── Zomato Restaurant reviews.csv
├── notebook/
│   └── Zomato_Restaurant_Clustering.ipynb
├── models/
│   ├── kmeans_zomato_model.pkl
│   └── scaler_zomato.pkl
├── outputs/
│   └── zomato_clustered_restaurants.csv
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place `Zomato Restaurant names and Metadata.csv` and `Zomato Restaurant reviews.csv` in the `data/` folder (or update the file paths at the top of the notebook).
3. Open and run `notebook/Zomato_Restaurant_Clustering.ipynb` top to bottom (if not using Google Colab, remove/skip the `google.colab.drive.mount` cell).
4. The notebook saves the trained model (`kmeans_zomato_model.pkl`), the fitted scaler (`scaler_zomato.pkl`), and the labeled dataset (`zomato_clustered_restaurants.csv`) at the end.

### Assigning a new restaurant to a cluster

```python
import pickle

with open('models/kmeans_zomato_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler_zomato.pkl', 'rb') as f:
    scaler = pickle.load(f)

# order: [Cost, num_cuisines, num_collections, avg_rating, rating_std, avg_pictures]
new_restaurant = [[cost, num_cuisines, num_collections, avg_rating, rating_std, avg_pictures]]
new_scaled = scaler.transform(new_restaurant)
predicted_cluster = model.predict(new_scaled)
```

## Tech Stack

Python, pandas, numpy, matplotlib, seaborn, scikit-learn (`StandardScaler`, `KMeans`, `AgglomerativeClustering`, `PCA`, `silhouette_score`, `adjusted_rand_score`).

## Author's Note

*"I aggregated review data per restaurant to avoid bias, selected k=3 for K-Means after comparing it against both a statistically 'better' k=2 and more granular k=7/k=8 alternatives, and found three data-driven segments that differ most clearly in price, curation, and rating consistency — while being careful to describe these as patterns in this dataset rather than universal quality judgments."*
