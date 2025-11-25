import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, kstest, expon, gamma, lognorm, beta, uniform
from scipy import stats
import numpy as np
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
base_path = r"C:\Users\Winte\OneDrive\Desktop\Spotify Data Collection 4\output_links"
emotion_files = {
    "happy": "happy_tracks.csv",
    "sad": "sad_tracks.csv",
    "energetic": "energetic_tracks.csv",
    "love": "love_tracks.csv"
}

# -------------------------------
# LOAD AND COMBINE DATA
# -------------------------------
dfs = []
for emotion, filename in emotion_files.items():
    file_path = os.path.join(base_path, filename)
    df = pd.read_csv(file_path)
    df['emotion'] = emotion
    dfs.append(df)

# Merge all into one DataFrame
all_tracks = pd.concat(dfs, ignore_index=True)

# -------------------------------
# PLOT DISTRIBUTIONS
# -------------------------------
plt.figure(figsize=(10,6))
sns.histplot(data=all_tracks, x='popularity', hue='emotion', bins=30, kde=True, common_norm=False)
plt.title("Popularity Distributions by Emotion")
plt.xlabel("Popularity (0–100)")
plt.ylabel("Count")
plt.savefig('popularity_distributions_by_emotion.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved as 'popularity_distributions_by_emotion.png'\n")
plt.close()

# -------------------------------
# NORMALITY TESTS PER EMOTION
# -------------------------------
print("Normality Test Results (D’Agostino & Pearson):\n")
for emotion in emotion_files.keys():
    subset = all_tracks[all_tracks['emotion'] == emotion]['popularity'].dropna()
    stat, p = normaltest(subset)
    print(f"{emotion.capitalize():<10}  |  p-value: {p:.5f}  |  "
          f"{'Reject normality ❌' if p < 0.05 else 'Likely normal ✅'}")

# -------------------------------
# OPTIONAL: DESCRIPTIVE STATS
# -------------------------------
desc = all_tracks.groupby('emotion')['popularity'].describe()[['mean','std','min','25%','50%','75%','max']]
print("\nDescriptive Statistics:")
print(desc.round(2))

# -------------------------------
# DISTRIBUTION IDENTIFICATION
# -------------------------------
print("\n" + "="*70)
print("BEST-FIT DISTRIBUTION ANALYSIS")
print("="*70)

distributions = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Gamma': stats.gamma,
    'Log-Normal': stats.lognorm,
    'Beta': stats.beta,
    'Uniform': stats.uniform,
    'Weibull': stats.weibull_min
}

for emotion in emotion_files.keys():
    subset = all_tracks[all_tracks['emotion'] == emotion]['popularity'].dropna().values
    
    print(f"\n{emotion.upper()} - Distribution Fit Results:")
    print("-" * 70)
    
    fit_results = {}
    
    for dist_name, distribution in distributions.items():
        try:
            # Fit distribution parameters
            params = distribution.fit(subset)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = kstest(subset, distribution.cdf, args=params)
            
            # Calculate AIC (lower is better)
            log_likelihood = np.sum(distribution.logpdf(subset, *params))
            k = len(params)  # number of parameters
            aic = 2 * k - 2 * log_likelihood
            
            fit_results[dist_name] = {
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'aic': aic,
                'params': params
            }
        except Exception as e:
            continue
    
    # Sort by AIC (lower is better)
    sorted_fits = sorted(fit_results.items(), key=lambda x: x[1]['aic'])
    
    print(f"{'Distribution':<15} {'KS Stat':<12} {'KS p-value':<12} {'AIC':<15} {'Fit Quality'}")
    print("-" * 70)
    
    for i, (dist_name, results) in enumerate(sorted_fits[:5]):  # Show top 5
        ks_stat = results['ks_stat']
        ks_p = results['ks_p']
        aic = results['aic']
        
        # Determine fit quality
        if ks_p > 0.05:
            quality = "✅ Good fit"
        elif ks_p > 0.01:
            quality = "⚠️ Marginal"
        else:
            quality = "❌ Poor fit"
        
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        
        print(f"{rank} {dist_name:<15} {ks_stat:<12.4f} {ks_p:<12.4f} {aic:<15.2f} {quality}")
    
    # Print best fit
    best_dist = sorted_fits[0][0]
    best_p = sorted_fits[0][1]['ks_p']
    print(f"\n→ Best fit: {best_dist} (KS p-value = {best_p:.4f})")

print("\n" + "="*70)
print("Note: Higher KS p-value (>0.05) and lower AIC indicate better fit")
print("="*70)
