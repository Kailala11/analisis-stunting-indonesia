"""
PROJECT DATA MINING: ANALISIS PATTERN STUNTING INDONESIA
========================================================

Dataset: SSGI (Survei Status Gizi Indonesia) 2021-2024
Sumber Inspirasi: Data Kementerian Kesehatan RI

TUJUAN PROJECT:
1. Pattern Discovery: Menemukan pola penurunan stunting
2. Clustering: Mengelompokkan provinsi berdasarkan karakteristik
3. Association Rules: Mencari hubungan faktor risiko dengan stunting
4. Predictive Analysis: Prediksi daerah yang butuh prioritas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DAN EKSPLORASI DATA
# ============================================================================

print("="*70)
print("PROJECT DATA MINING: ANALISIS PATTERN STUNTING INDONESIA")
print("="*70)

# Load data
df = pd.read_csv('data_stunting_provinsi.csv')

print("\n[1] DATA OVERVIEW")
print("-" * 70)
print(f"Jumlah Provinsi: {len(df)}")
print(f"Periode Data: 2021-2024")
print("\nContoh Data:")
print(df.head())

print("\n[2] STATISTIK DESKRIPTIF")
print("-" * 70)
print(df.describe())

# ============================================================================
# 2. PATTERN DISCOVERY: ANALISIS TREN PENURUNAN STUNTING
# ============================================================================

print("\n" + "="*70)
print("[3] PATTERN DISCOVERY: TREND ANALYSIS")
print("="*70)

# Hitung penurunan stunting per provinsi
df['Penurunan_2021_2024'] = df['Prevalensi_2021'] - df['Prevalensi_2024']
df['Persen_Penurunan'] = (df['Penurunan_2021_2024'] / df['Prevalensi_2021']) * 100

# Top 10 provinsi dengan penurunan terbesar
print("\nTOP 10 Provinsi dengan Penurunan Stunting Terbesar (2021-2024):")
top_performers = df.nlargest(10, 'Penurunan_2021_2024')[['Provinsi', 'Prevalensi_2021', 'Prevalensi_2024', 'Penurunan_2021_2024', 'Persen_Penurunan']]
print(top_performers.to_string(index=False))

# Provinsi dengan penurunan terkecil / bahkan naik
print("\nProvinsi dengan Penurunan Terkecil:")
worst_performers = df.nsmallest(5, 'Penurunan_2021_2024')[['Provinsi', 'Prevalensi_2021', 'Prevalensi_2024', 'Penurunan_2021_2024']]
print(worst_performers.to_string(index=False))

# Analisis per region
print("\nRata-rata Prevalensi Stunting per Region (2024):")
regional_stats = df.groupby('Region').agg({
    'Prevalensi_2024': 'mean',
    'Penurunan_2021_2024': 'mean',
    'Tingkat_Kemiskinan': 'mean'
}).round(2)
print(regional_stats)

# ============================================================================
# 3. CLUSTERING: PENGELOMPOKAN PROVINSI
# ============================================================================

print("\n" + "="*70)
print("[4] CLUSTERING ANALYSIS: Pengelompokan Provinsi")
print("="*70)

# Pilih fitur untuk clustering
features_for_clustering = ['Prevalensi_2024', 'Cakupan_Imunisasi', 'Cakupan_ASI_Eksklusif', 
                           'Akses_Air_Bersih', 'Akses_Sanitasi', 'Tingkat_Kemiskinan']

X = df[features_for_clustering]

# Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering dengan 4 cluster
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Label cluster berdasarkan karakteristik
cluster_labels = {
    0: 'Cluster A',
    1: 'Cluster B', 
    2: 'Cluster C',
    3: 'Cluster D'
}

print("\nHasil Clustering:")
for cluster_id in range(4):
    cluster_provinces = df[df['Cluster'] == cluster_id]['Provinsi'].tolist()
    avg_stunting = df[df['Cluster'] == cluster_id]['Prevalensi_2024'].mean()
    avg_poverty = df[df['Cluster'] == cluster_id]['Tingkat_Kemiskinan'].mean()
    
    print(f"\n{cluster_labels[cluster_id]} (Avg Stunting: {avg_stunting:.1f}%, Avg Kemiskinan: {avg_poverty:.1f}%):")
    print(f"  Provinsi: {', '.join(cluster_provinces)}")

print("\nKarakteristik Setiap Cluster:")
cluster_chars = df.groupby('Cluster')[features_for_clustering].mean().round(2)
print(cluster_chars)

# ============================================================================
# 4. CORRELATION ANALYSIS: FAKTOR RISIKO STUNTING
# ============================================================================

print("\n" + "="*70)
print("[5] CORRELATION ANALYSIS: Faktor Risiko Stunting")
print("="*70)

# Korelasi dengan stunting 2024
correlations = df[['Prevalensi_2024', 'Cakupan_Imunisasi', 'Cakupan_ASI_Eksklusif',
                   'Akses_Air_Bersih', 'Akses_Sanitasi', 'Tingkat_Kemiskinan']].corr()['Prevalensi_2024'].sort_values(ascending=False)

print("\nKorelasi Faktor dengan Prevalensi Stunting 2024:")
print(correlations)

# Interpretasi
print("\nINTERPRETASI:")
print("- Korelasi positif: faktor yang meningkat, stunting juga meningkat")
print("- Korelasi negatif: faktor yang meningkat, stunting menurun")
print("\nFaktor dengan korelasi negatif kuat (baik untuk turunkan stunting):")
for factor, corr in correlations.items():
    if corr < -0.5 and factor != 'Prevalensi_2024':
        print(f"  • {factor}: {corr:.3f}")

# ============================================================================
# 5. ASSOCIATION RULES: PATTERN MINING
# ============================================================================

print("\n" + "="*70)
print("[6] PATTERN MINING: Association Rules")
print("="*70)

# Kategorisasi variabel untuk association rules
df['Stunting_Category'] = pd.cut(df['Prevalensi_2024'], 
                                  bins=[0, 15, 20, 30, 100],
                                  labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'])

df['Imunisasi_Category'] = pd.cut(df['Cakupan_Imunisasi'],
                                   bins=[0, 65, 75, 100],
                                   labels=['Rendah', 'Sedang', 'Tinggi'])

df['Kemiskinan_Category'] = pd.cut(df['Tingkat_Kemiskinan'],
                                    bins=[0, 7, 12, 30],
                                    labels=['Rendah', 'Sedang', 'Tinggi'])

df['Sanitasi_Category'] = pd.cut(df['Akses_Sanitasi'],
                                  bins=[0, 75, 85, 100],
                                  labels=['Rendah', 'Sedang', 'Tinggi'])

print("\nPola yang Ditemukan:")
print("\n1. Stunting Tinggi + Kemiskinan Tinggi:")
pattern1 = df[(df['Stunting_Category'].isin(['Tinggi', 'Sangat Tinggi'])) & 
              (df['Kemiskinan_Category'] == 'Tinggi')]
print(f"   Jumlah provinsi: {len(pattern1)}")
print(f"   Provinsi: {', '.join(pattern1['Provinsi'].tolist())}")

print("\n2. Stunting Rendah + Imunisasi Tinggi + Sanitasi Tinggi:")
pattern2 = df[(df['Stunting_Category'] == 'Rendah') & 
              (df['Imunisasi_Category'] == 'Tinggi') & 
              (df['Sanitasi_Category'] == 'Tinggi')]
print(f"   Jumlah provinsi: {len(pattern2)}")
print(f"   Provinsi: {', '.join(pattern2['Provinsi'].tolist())}")

print("\n3. Stunting Tinggi + Imunisasi Rendah:")
pattern3 = df[(df['Stunting_Category'].isin(['Tinggi', 'Sangat Tinggi'])) & 
              (df['Imunisasi_Category'] == 'Rendah')]
print(f"   Jumlah provinsi: {len(pattern3)}")
print(f"   Provinsi: {', '.join(pattern3['Provinsi'].tolist())}")

# ============================================================================
# 6. PREDICTIVE ANALYSIS: PRIORITAS INTERVENSI
# ============================================================================

print("\n" + "="*70)
print("[7] PREDICTIVE ANALYSIS & PRIORITAS INTERVENSI")
print("="*70)

# Prediksi stunting 2025 berdasarkan tren
df['Tren_Penurunan_Per_Tahun'] = df['Penurunan_2021_2024'] / 3
df['Prediksi_2025'] = df['Prevalensi_2024'] - df['Tren_Penurunan_Per_Tahun']

# Scoring untuk prioritas intervensi
# Skor tinggi = butuh prioritas tinggi
df['Priority_Score'] = (
    df['Prevalensi_2024'] * 0.4 +  # Prevalensi saat ini
    (100 - df['Cakupan_Imunisasi']) * 0.2 +  # Gap imunisasi
    df['Tingkat_Kemiskinan'] * 0.2 +  # Kemiskinan
    (100 - df['Akses_Sanitasi']) * 0.2  # Gap sanitasi
)

print("\nTOP 10 Provinsi Prioritas Tinggi (Butuh Intervensi Segera):")
priority_provinces = df.nlargest(10, 'Priority_Score')[
    ['Provinsi', 'Prevalensi_2024', 'Prediksi_2025', 'Priority_Score', 
     'Cakupan_Imunisasi', 'Tingkat_Kemiskinan']
].round(2)
print(priority_provinces.to_string(index=False))

print("\nRekomendasi Intervensi per Cluster:")
for cluster_id in range(4):
    cluster_data = df[df['Cluster'] == cluster_id]
    avg_immun = cluster_data['Cakupan_Imunisasi'].mean()
    avg_sanit = cluster_data['Akses_Sanitasi'].mean()
    avg_asi = cluster_data['Cakupan_ASI_Eksklusif'].mean()
    
    print(f"\n{cluster_labels[cluster_id]}:")
    
    if avg_immun < 70:
        print("  • PRIORITAS: Tingkatkan program imunisasi")
    if avg_sanit < 80:
        print("  • PRIORITAS: Perbaiki akses sanitasi")
    if avg_asi < 75:
        print("  • PRIORITAS: Kampanye ASI eksklusif")

# ============================================================================
# 7. VISUALISASI
# ============================================================================

print("\n" + "="*70)
print("[8] Membuat Visualisasi...")
print("="*70)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Trend Line Chart
ax1 = plt.subplot(2, 3, 1)
years = ['2021', '2022', '2023', '2024']
top5 = df.nlargest(5, 'Prevalensi_2024')
for idx, row in top5.iterrows():
    values = [row['Prevalensi_2021'], row['Prevalensi_2022'], 
              row['Prevalensi_2023'], row['Prevalensi_2024']]
    ax1.plot(years, values, marker='o', label=row['Provinsi'])
ax1.set_title('Trend Stunting: Top 5 Provinsi Tertinggi', fontsize=12, fontweight='bold')
ax1.set_xlabel('Tahun')
ax1.set_ylabel('Prevalensi (%)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Bar Chart - Penurunan Terbesar
ax2 = plt.subplot(2, 3, 2)
top10_drop = df.nlargest(10, 'Penurunan_2021_2024')
ax2.barh(range(len(top10_drop)), top10_drop['Penurunan_2021_2024'])
ax2.set_yticks(range(len(top10_drop)))
ax2.set_yticklabels(top10_drop['Provinsi'], fontsize=8)
ax2.set_xlabel('Penurunan Stunting (%)')
ax2.set_title('Top 10 Penurunan Stunting Terbesar (2021-2024)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Scatter Plot - Kemiskinan vs Stunting
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(df['Tingkat_Kemiskinan'], df['Prevalensi_2024'], 
                     c=df['Cluster'], cmap='viridis', s=100, alpha=0.6)
ax3.set_xlabel('Tingkat Kemiskinan (%)')
ax3.set_ylabel('Prevalensi Stunting 2024 (%)')
ax3.set_title('Korelasi: Kemiskinan vs Stunting', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Cluster')
ax3.grid(True, alpha=0.3)

# 4. Box Plot - Regional Comparison
ax4 = plt.subplot(2, 3, 4)
df.boxplot(column='Prevalensi_2024', by='Region', ax=ax4)
ax4.set_xlabel('Region')
ax4.set_ylabel('Prevalensi Stunting 2024 (%)')
ax4.set_title('Distribusi Stunting per Region', fontsize=12, fontweight='bold')
plt.suptitle('')  # Remove default title
ax4.tick_params(axis='x', rotation=45)

# 5. Correlation Heatmap
ax5 = plt.subplot(2, 3, 5)
corr_features = ['Prevalensi_2024', 'Cakupan_Imunisasi', 'Cakupan_ASI_Eksklusif',
                 'Akses_Air_Bersih', 'Akses_Sanitasi', 'Tingkat_Kemiskinan']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax5, 
            cbar_kws={'label': 'Correlation'})
ax5.set_title('Correlation Matrix: Faktor Risiko Stunting', fontsize=12, fontweight='bold')

# 6. Cluster Visualization (PCA)
ax6 = plt.subplot(2, 3, 6)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
scatter2 = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], 
                       cmap='viridis', s=100, alpha=0.6)
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax6.set_title('Clustering Visualization (PCA)', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax6, label='Cluster')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_stunting_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi disimpan: analisis_stunting_visualization.png")

# ============================================================================
# 8. RINGKASAN & REKOMENDASI
# ============================================================================

print("\n" + "="*70)
print("RINGKASAN & REKOMENDASI KEBIJAKAN")
print("="*70)

print("\n📊 KEY FINDINGS:")
print(f"1. Prevalensi stunting nasional 2024: {df['Prevalensi_2024'].mean():.1f}%")
print(f"2. Penurunan rata-rata 2021-2024: {df['Penurunan_2021_2024'].mean():.1f}%")
print(f"3. Region dengan stunting tertinggi: {regional_stats['Prevalensi_2024'].idxmax()}")
print(f"4. Region dengan stunting terendah: {regional_stats['Prevalensi_2024'].idxmin()}")

print("\n🎯 REKOMENDASI STRATEGIS:")
print("1. FOKUS GEOGRAFIS:")
print(f"   • Prioritaskan intervensi di: {', '.join(priority_provinces.head(3)['Provinsi'].tolist())}")

print("\n2. INTERVENSI BERBASIS CLUSTER:")
worst_cluster = cluster_chars['Prevalensi_2024'].idxmax()
print(f"   • Cluster dengan stunting tertinggi: {cluster_labels[worst_cluster]}")
print(f"   • Fokus pada: peningkatan imunisasi dan akses sanitasi")

print("\n3. FAKTOR KUNCI:")
top_negative_corr = correlations[correlations < -0.5]
top_negative_corr = top_negative_corr[top_negative_corr.index != 'Prevalensi_2024']
if len(top_negative_corr) > 0:
    print("   • Faktor protektif yang harus ditingkatkan:")
    for factor in top_negative_corr.index:
        print(f"     - {factor}")

print("\n4. TARGET 2025:")
print(f"   • Prediksi stunting nasional 2025: {df['Prediksi_2025'].mean():.1f}%")
print(f"   • Target pemerintah: 18.8%")
gap = df['Prediksi_2025'].mean() - 18.8
if gap > 0:
    print(f"   • Gap yang harus dipercepat: {gap:.1f}%")

print("\n" + "="*70)
print("ANALISIS SELESAI!")
print("="*70)
print("\nFile output:")
print("• analisis_stunting_visualization.png")
print("• data_stunting_provinsi.csv (dengan cluster & prediksi)")

# Save enriched data
df.to_csv('data_stunting_enriched.csv', index=False)
print("• data_stunting_enriched.csv (data + hasil analisis)")

print("\n✨ Project data mining stunting berhasil diselesaikan!")
