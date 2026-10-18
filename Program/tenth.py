import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# --- Reproducibility ---
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Visualization & Pandas Settings ---
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)

# --- File Loading (Local PyCharm Runtime) ---
csv_candidates = [
    f for f in os.listdir(".")
    if f.lower().endswith(".csv")
]

if not csv_candidates:
    raise FileNotFoundError(
        "Place your placement CSV in the project directory "
        "or specify DATA_PATH explicitly."
    )

DATA_PATH = csv_candidates[0]

# --- Load Data ---
df = pd.read_csv(DATA_PATH)

print("Dataset path:", DATA_PATH)
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.info()

# --- Numerical and categorical columns ---
numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical columns:")
print(numeric_cols_all)

print("\nCategorical columns:")
print(categorical_cols)

print("\nMissing values:")
print(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame("missing_count")
    .head(20)
)

# --- Columns excluded from unsupervised learning ---
EXCLUDE_FROM_UNSUPERVISED = [
    "StudentID",
    "PlacementStatus",
    "Salary Package",
    "IsAnomaly"
]

missing_excluded = [
    c for c in EXCLUDE_FROM_UNSUPERVISED
    if c not in df.columns
]

if missing_excluded:
    print(
        "\nColumns not found (safe to ignore):",
        missing_excluded
    )

available_excluded = [
    c for c in EXCLUDE_FROM_UNSUPERVISED
    if c in df.columns
]

print(
    "\nExcluded from clustering:",
    available_excluded
)

# --- Useful audit: relationship of candidate numeric variables with target ---
if "PlacementStatus" in df.columns:

    corr = (
        df[numeric_cols_all]
        .corr()["PlacementStatus"]
        .sort_values(ascending=False)
    )

    print("\nCorrelation with PlacementStatus:")
    print(corr.to_frame("corr_with_PlacementStatus"))

# --- Modeling features ---
FEATURES = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]

FEATURES = [
    c for c in FEATURES
    if c in df.columns
]

print("\nNumber of modeling features:", len(FEATURES))
print(FEATURES)

X_raw = df[FEATURES].copy()

y = (
    df["PlacementStatus"].copy()
    if "PlacementStatus" in df.columns
    else None
)

print("\nSummary Statistics of Raw Features:")
print(X_raw.describe().T)

# --- Exploratory preprocessing for unsupervised experiments ---
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_scaled = preprocess.fit_transform(X_raw)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=FEATURES,
    index=df.index
)

print("\nProcessed matrix shape:")
print(X_scaled.shape)

print("\nScaled Data Head:")
print(X_scaled.head())

print("\nMeans after standardisation (approximately 0):")
print(
    X_scaled.mean()
    .round(3)
    .to_frame("mean")
)

print("\nStd after standardisation (approximately 1):")
print(
    X_scaled.std(ddof=0)
    .round(3)
    .to_frame("std")
)

# ============================================================
# PCA VISUALIZATION
# ============================================================

pca_vis = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca2 = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca2[:, 0],
    X_pca2[:, 1],
    s=8,
    alpha=0.35
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Student Data Projected onto First Two Principal Components"
)

plt.show()

print(
    "Variance explained by PC1 + PC2:",
    round(
        pca_vis.explained_variance_ratio_.sum() * 100,
        2
    ),
    "%"
)

# ============================================================
# K-MEANS CLUSTERING EVALUATION
# ============================================================

K_VALUES = list(range(2, 9))

results_kmeans = []

for k in K_VALUES:

    print(f"Evaluating KMeans for K={k}...")

    # --- Random initialization ---
    km_random = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels_random = km_random.fit_predict(X_scaled)

    # --- K-Means++ initialization ---
    km_plus = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels_plus = km_plus.fit_predict(X_scaled)

    results_kmeans.append({
        "K": k,

        "Random_Inertia":
            km_random.inertia_,

        "Random_Silhouette":
            silhouette_score(
                X_scaled,
                labels_random
            ),

        "KMeans++_Inertia":
            km_plus.inertia_,

        "KMeans++_Silhouette":
            silhouette_score(
                X_scaled,
                labels_plus
            )
    })

kmeans_results = pd.DataFrame(
    results_kmeans
)

print("\nK-Means Results:")
print(
    kmeans_results.round(4)
)

# ============================================================
# ELBOW METHOD
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    kmeans_results["K"],
    kmeans_results["Random_Inertia"],
    marker="o",
    label="Random init"
)

ax.plot(
    kmeans_results["K"],
    kmeans_results["KMeans++_Inertia"],
    marker="o",
    label="K-Means++"
)

ax.set_xlabel(
    "Number of clusters (K)"
)

ax.set_ylabel(
    "Inertia"
)

ax.set_title(
    "Elbow Method: Inertia vs K"
)

ax.legend()

plt.show()

# ============================================================
# SILHOUETTE SCORE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    kmeans_results["K"],
    kmeans_results["Random_Silhouette"],
    marker="o",
    label="Random init"
)

ax.plot(
    kmeans_results["K"],
    kmeans_results["KMeans++_Silhouette"],
    marker="o",
    label="K-Means++"
)

ax.set_xlabel(
    "Number of clusters (K)"
)

ax.set_ylabel(
    "Silhouette score"
)

ax.set_title(
    "Silhouette Score vs K"
)

ax.axhline(
    0,
    linewidth=0.8
)

ax.legend()

plt.show()

# --- Best K based on K-Means++ silhouette ---
best_k_silhouette = int(
    kmeans_results.loc[
        kmeans_results["KMeans++_Silhouette"].idxmax(),
        "K"
    ]
)

print(
    "K with highest K-Means++ silhouette:",
    best_k_silhouette
)

K_BEST = best_k_silhouette

# ============================================================
# FINAL K-MEANS MODELS
# ============================================================

km_random_final = KMeans(
    n_clusters=K_BEST,
    init="random",
    n_init=10,
    random_state=RANDOM_STATE
)

labels_random_final = km_random_final.fit_predict(
    X_scaled
)

km_plus_final = KMeans(
    n_clusters=K_BEST,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE
)

labels_plus_final = km_plus_final.fit_predict(
    X_scaled
)

# ============================================================
# K-MEANS COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "K-Means random",
        "K-Means++"
    ],

    "K": [
        K_BEST,
        K_BEST
    ],

    "Inertia": [
        km_random_final.inertia_,
        km_plus_final.inertia_
    ],

    "Silhouette": [
        silhouette_score(
            X_scaled,
            labels_random_final
        ),

        silhouette_score(
            X_scaled,
            labels_plus_final
        )
    ],

    "Davies_Bouldin": [
        davies_bouldin_score(
            X_scaled,
            labels_random_final
        ),

        davies_bouldin_score(
            X_scaled,
            labels_plus_final
        )
    ]
})

display(
    comparison.round(4)
)

# ============================================================
# DBSCAN SETUP
# ============================================================

MIN_SAMPLES_VALUES = [
    5,
    10,
    15,
    20
]

EPS_VALUES = np.arange(
    0.5,
    2.1,
    0.1
)

# --- DBSCAN exploration sample ---
DB_SAMPLE_SIZE = min(
    5000,
    len(X_scaled)
)

dbscan_rng = np.random.default_rng(
    RANDOM_STATE
)

db_idx = dbscan_rng.choice(
    len(X_scaled),
    size=DB_SAMPLE_SIZE,
    replace=False
)

X_db_sample = X_scaled.iloc[
    db_idx
].to_numpy()

print(
    "\nDBSCAN exploration sample:",
    X_db_sample.shape
)

# ============================================================
# DBSCAN SENSITIVITY EVALUATION
# ============================================================

dbscan_results = []

for min_samples in MIN_SAMPLES_VALUES:

    for eps in EPS_VALUES:

        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = db.fit_predict(
            X_db_sample
        )

        n_clusters = (
            len(set(labels))
            - (1 if -1 in labels else 0)
        )

        noise_pct = (
            (labels == -1).mean()
            * 100
        )

        # --- Silhouette on non-noise points ---
        non_noise_mask = labels != -1

        non_noise_labels = labels[
            non_noise_mask
        ]

        if (
            n_clusters >= 2
            and len(set(non_noise_labels)) >= 2
        ):

            silhouette = silhouette_score(
                X_db_sample[non_noise_mask],
                non_noise_labels
            )

        else:

            silhouette = np.nan

        dbscan_results.append({

            "eps": eps,

            "min_samples": min_samples,

            "clusters": n_clusters,

            "noise_pct": noise_pct,

            "silhouette_non_noise": silhouette
        })

dbscan_results = pd.DataFrame(
    dbscan_results
)

print("\nDBSCAN Results:")

print(
    dbscan_results.round(4)
)

# ============================================================
# DBSCAN NOISE SENSITIVITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

for m in MIN_SAMPLES_VALUES:

    sub = dbscan_results[
        dbscan_results["min_samples"] == m
    ]

    ax.plot(
        sub["eps"],
        sub["noise_pct"],
        marker="o",
        label=f"min_samples={m}"
    )

ax.set_xlabel(
    "eps"
)

ax.set_ylabel(
    "Noise percentage"
)

ax.set_title(
    "DBSCAN Sensitivity — Noise Percentage"
)

ax.legend()

plt.show()

# ============================================================
# DBSCAN SILHOUETTE SENSITIVITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

for m in MIN_SAMPLES_VALUES:

    sub = dbscan_results[
        dbscan_results["min_samples"] == m
    ]

    ax.plot(
        sub["eps"],
        sub["silhouette_non_noise"],
        marker="o",
        label=f"min_samples={m}"
    )

ax.set_xlabel(
    "eps"
)

ax.set_ylabel(
    "Silhouette score"
)

ax.set_title(
    "DBSCAN Sensitivity — Silhouette Score"
)

ax.legend()

plt.show()

# ============================================================
# SELECT BEST DBSCAN PARAMETERS
# ============================================================

valid_db = dbscan_results[
    dbscan_results[
        "silhouette_non_noise"
    ].notna()
    &
    (
        dbscan_results[
            "noise_pct"
        ] < 50
    )
].copy()

if len(valid_db):

    best_db_row = valid_db.sort_values(
        [
            "silhouette_non_noise",
            "noise_pct"
        ],
        ascending=[
            False,
            True
        ]
    ).iloc[0]

    BEST_EPS = float(
        best_db_row["eps"]
    )

    BEST_MIN_SAMPLES = int(
        best_db_row["min_samples"]
    )

else:

    BEST_EPS = 1.2
    BEST_MIN_SAMPLES = 10

print(
    "\nCandidate DBSCAN parameters:"
)

print(
    "eps =",
    BEST_EPS
)

print(
    "min_samples =",
    BEST_MIN_SAMPLES
)

# ============================================================
# FINAL DBSCAN ON EXPLORATION SAMPLE
# ============================================================

db_final_sample = DBSCAN(
    eps=BEST_EPS,
    min_samples=BEST_MIN_SAMPLES,
    n_jobs=-1
).fit_predict(
    X_db_sample
)

# --- PCA visualization ---
db_pca2 = PCA(
    n_components=2,
    random_state=RANDOM_STATE
).fit_transform(
    X_db_sample
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    db_pca2[:, 0],
    db_pca2[:, 1],
    c=db_final_sample,
    s=10,
    alpha=0.5,
    cmap="tab10"
)

plt.xlabel(
    "PCA PC1"
)

plt.ylabel(
    "PCA PC2"
)

plt.title(
    f"DBSCAN on Exploration Sample: "
    f"eps={BEST_EPS}, "
    f"min_samples={BEST_MIN_SAMPLES}"
)

plt.show()

print(
    "Clusters:",
    len(set(db_final_sample))
    - (1 if -1 in db_final_sample else 0)
)

print(
    "Noise points:",
    int(
        (db_final_sample == -1).sum()
    )
)

print(
    "Noise percentage:",
    round(
        (db_final_sample == -1).mean() * 100,
        2
    ),
    "%"
)

# ============================================================
# OPTIONAL FULL DATA DBSCAN
# ============================================================

RUN_FULL_DBSCAN = False

if RUN_FULL_DBSCAN:

    db_full = DBSCAN(
        eps=BEST_EPS,
        min_samples=BEST_MIN_SAMPLES,
        n_jobs=-1
    )

    db_full_labels = db_full.fit_predict(
        X_scaled.to_numpy()
    )

    print(
        "Full-data DBSCAN clusters:",
        len(set(db_full_labels))
        - (1 if -1 in db_full_labels else 0)
    )

    print(
        "Full-data noise points:",
        int(
            (db_full_labels == -1).sum()
        )
    )

    print(
        "Full-data noise percentage:",
        round(
            (db_full_labels == -1).mean() * 100,
            2
        ),
        "%"
    )

else:

    print(
        "Full-data DBSCAN is OFF. "
        "Set RUN_FULL_DBSCAN=True to run it."
    )

# ============================================================
# HIERARCHICAL CLUSTERING SAMPLE
# ============================================================

HIER_SAMPLE_SIZE = min(
    2000,
    len(X_scaled)
)

hier_idx = dbscan_rng.choice(
    len(X_scaled),
    size=HIER_SAMPLE_SIZE,
    replace=False
)

X_hier_sample = X_scaled.iloc[
    hier_idx
].to_numpy()

print(
    "Hierarchical sample:",
    X_hier_sample.shape
)

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

print(
    "\nRunning hierarchical clustering..."
)

hier_linkage = linkage(
    X_hier_sample,
    method="ward"
)

plt.figure(
    figsize=(12, 6)
)

dendrogram(
    hier_linkage,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=9
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel(
    "Cluster / Sample"
)

plt.ylabel(
    "Distance"
)

plt.show()

# ============================================================
# AGGLOMERATIVE CLUSTERING
# ============================================================

HIER_K = K_BEST

hier_model = AgglomerativeClustering(
    n_clusters=HIER_K,
    linkage="ward"
)

hier_labels = hier_model.fit_predict(
    X_hier_sample
)

print(
    "\nHierarchical clustering clusters:",
    len(np.unique(hier_labels))
)

# ============================================================
# HIERARCHICAL CLUSTERING METRICS
# ============================================================

hier_silhouette = silhouette_score(
    X_hier_sample,
    hier_labels
)

hier_db = davies_bouldin_score(
    X_hier_sample,
    hier_labels
)

hier_ch = calinski_harabasz_score(
    X_hier_sample,
    hier_labels
)

print(
    "\nHierarchical Clustering Metrics:"
)

print(
    "Silhouette Score:",
    round(hier_silhouette, 4)
)

print(
    "Davies-Bouldin Score:",
    round(hier_db, 4)
)

print(
    "Calinski-Harabasz Score:",
    round(hier_ch, 4)
)

# ============================================================
# PCA VISUALIZATION OF HIERARCHICAL CLUSTERS
# ============================================================

hier_pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_hier_pca = hier_pca.fit_transform(
    X_hier_sample
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    X_hier_pca[:, 0],
    X_hier_pca[:, 1],
    c=hier_labels,
    s=12,
    alpha=0.6,
    cmap="tab10"
)

plt.xlabel(
    "PCA PC1"
)

plt.ylabel(
    "PCA PC2"
)

plt.title(
    f"Hierarchical Clustering "
    f"(K={HIER_K})"
)

plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("UNSUPERVISED CLUSTERING ANALYSIS COMPLETED")
print("=" * 60)

print(
    "\nDataset:",
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)

print(
    "Number of features:",
    len(FEATURES)
)

print(
    "Best K-Means K:",
    K_BEST
)

print(
    "Best DBSCAN eps:",
    BEST_EPS
)

print(
    "Best DBSCAN min_samples:",
    BEST_MIN_SAMPLES
)

print(
    "Hierarchical K:",
    HIER_K
)

print(
    "\nAnalysis includes:"
)

print("1. Data loading and inspection")
print("2. Missing-value analysis")
print("3. Feature preprocessing")
print("4. Standardisation")
print("5. PCA visualization")
print("6. K-Means with random initialization")
print("7. K-Means++")
print("8. Elbow method")
print("9. Silhouette analysis")
print("10. DBSCAN parameter sensitivity")
print("11. DBSCAN clustering")
print("12. Hierarchical clustering")
print("13. Dendrogram")
print("14. Clustering evaluation metrics")

print("\nDone.")