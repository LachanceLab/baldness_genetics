"""
This code computes the power required for replication of GWAS hits using population-specific allele 
frequencies and effect sizes from Heilmann-Haimbach et. al. [1]

Genotype Risk Ratio is calculated from Effect sizes (beta) using equation 2 from So et. al.[2]:
GRR = OR/(1+faa(OR-1))
    where OR = e^beta, and faa is the penetrance of genotype aa (where a is non-effect allele)

The equations used for the power calculations are from Skol et. al. [3], as implemented in the web 
application https://csg.sph.umich.edu/abecasis/cats/gas_power_calculator/index.html


References:
1. Heilmann-Heimbach, Stefanie, et al. "Meta-analysis identifies novel risk loci and yields systematic 
   insights into the biology of male-pattern baldness." Nature communications 8.1 (2017): 14694.
2. So, Hon‐Cheong, et al. "Evaluating the heritability explained by known susceptibility variants: 
   a survey of ten complex diseases." Genetic epidemiology 35.5 (2011): 310-317.
3. Skol, Andrew D., et al. "Joint analysis is more efficient than replication-based analysis for 
   two-stage genome-wide association studies." Nature genetics 38.2 (2006): 209-213.

"""

import math
from scipy.stats import norm
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the additive model and return the power list
def calculate_additive_model(RAF_list, GRR_list, prev, alpha, cases, controls):
    #C is the lower confidence interval at alpha/2 significance threshold (for a two-tail test on standard normal distribution); alpha is probability
    # C = -norm.ppf(alpha)
    C=norm.ppf(alpha/2)
    # C=norm.ppf(1-alpha/2)
    results = {'Power_list': []}

    for RAF, GRR in zip(RAF_list, GRR_list):
        AAfreq = RAF ** 2
        ABfreq = 2 * RAF * (1 - RAF)
        BBfreq = (1 - RAF) ** 2

        x0, x1, x2 = 2.0 * GRR - 1.0, GRR, 1.0 #Additive model
        # x0, x1, x2 = GRR * GRR, GRR, 1.0 #Multiplicative model - different GRR
        denom = (x0 * AAfreq) + (x1 * ABfreq) + (x2 * BBfreq)

        AAprob = x0 * prev / denom
        ABprob = x1 * prev / denom
        BBprob = x2 * prev / denom

        casesRAF = ((AAprob * AAfreq) + (ABprob * ABfreq * 0.5)) / prev
        controlsRAF = (((1 - AAprob) * AAfreq) + ((1 - ABprob) * ABfreq * 0.5)) / (1 - prev)

        Vcases = casesRAF * (1 - casesRAF)
        Vcontrols = controlsRAF * (1 - controlsRAF)

        ncp = (casesRAF - controlsRAF) / math.sqrt(((Vcases / cases) + (Vcontrols / controls)) * 0.5)
        P=norm.cdf(C-ncp)+1-norm.cdf(-C-ncp)
        results['Power_list'].append(P)

    return results

# Count SNPs with enough power for different -log10(alpha) thresholds
def count_snps_with_power_above_threshold(RAF_list, GRR_list, prev, cases, controls):
    thresholds = np.arange(0.5, 5.5, 0.5)
    snp_counts = []

    for threshold in thresholds:
        alpha = 10**(-threshold)
        # print(alpha)
        results = calculate_additive_model(RAF_list, GRR_list, prev, alpha, cases, controls)
        power_list = results['Power_list']
        print(power_list)
        count = round(sum(power_list))
        snp_counts.append(count)

    return thresholds, snp_counts

cases = 990
controls = 1143
prev = 0.4650

HnH = pd.read_csv('HnHstatsinUKBBMADCaP.csv', sep=",")
HnH_Beta=list(HnH['Heilmann_Effect_Size_Risk_increasing_Allele'])

# MADCaP Data
MADCaP_RAF = list(HnH['MADCaP_HnH_Risk_increasing_allele_AF'])
MADCaP_RAF_controls = list(HnH['MADCaP_HnH_Risk_increasing_allele_Control_AF'])


MADCaP_RAF_cases = [(MADCaP_RAF[q] - ((1-prev)*MADCaP_RAF_controls[q]))/prev for q in range(len(MADCaP_RAF))]
MADCaP_faa=[prev*((1 - MADCaP_RAF_cases[q])**2)/((1 - MADCaP_RAF[q])**2) for q in range(len(MADCaP_RAF))]
MADCaP_GRR = [np.exp(HnH_Beta[q]) / (1 - MADCaP_faa[q] + MADCaP_faa[q] * np.exp(HnH_Beta[q])) for q in range(len(MADCaP_RAF))]


# UKBB Data
UKBB_RAF = list(HnH['UKBB_downsampled_HnH_Risk_increasing_allele_AF'])
UKBB_RAF_controls = list(HnH['UKBB_downsampled_HnH_Risk_increasing_allele_Control_AF'])

UKBB_RAF_cases = [(UKBB_RAF[q] - ((1-prev)*UKBB_RAF_controls[q]))/prev for q in range(len(UKBB_RAF))]
UKBB_faa=[prev*((1 - UKBB_RAF_cases[q])**2)/((1 - UKBB_RAF[q])**2) for q in range(len(UKBB_RAF))]
UKBB_GRR = [np.exp(HnH_Beta[q]) / (1 - UKBB_faa[q] + UKBB_faa[q] * np.exp(HnH_Beta[q])) for q in range(len(UKBB_RAF))]

# Count SNPs with enough power for different thresholds
MADCaP_thresholds, MADCaP_snp_counts = count_snps_with_power_above_threshold(MADCaP_RAF, MADCaP_GRR, prev, cases, controls)
UKBB_thresholds, UKBB_snp_counts = count_snps_with_power_above_threshold(UKBB_RAF, UKBB_GRR, prev, cases, controls)


UKBB_actual=[25,18,14,12,4,3,2,1,1,1]
MADCaP_actual=[14,5,3,2,0,0,0,0,0,0]
Null_thrudist=[13.28156617,4.2,1.328156617,0.42,0.132815662,0.042,0.013281566,0.0042,0.001328157,0.00042]
Null=[round(m) for m in Null_thrudist]


replabel_empirical="Observed"
replabel_power="Expected from power calculations"
nulllabel="Expected if no replication"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# MADCaP Plot

ax1.plot(UKBB_thresholds, UKBB_snp_counts, marker='s', color='orange', linestyle='dotted', label=replabel_power + ' in UKBB')
ax1.plot(MADCaP_thresholds, MADCaP_snp_counts, marker='s', color='green', linestyle='dotted', label=replabel_power + ' in MADCaP')
ax1.plot(MADCaP_thresholds, Null, marker='s', color='blue', linestyle='dotted', label=nulllabel)
ax1.set_xlabel('-log10(p-value) threshold in replication cohort', fontsize=10)
ax1.set_ylabel('Number of SNPs', fontsize=10)
# ax1.set_title('Replication in MADCaP', fontsize=12)
ax1.set_ybound(lower=-2, upper=42)
ax1.legend()

# UKBB Plot
ax2.plot(UKBB_thresholds, UKBB_actual, marker='o', color='orange', label=replabel_empirical + ' in UKBB')
ax2.plot(MADCaP_thresholds, MADCaP_actual, marker='o', color='green', label=replabel_empirical + ' in MADCaP')
ax2.plot(UKBB_thresholds, Null, marker='s', color='blue', linestyle='dotted', label=nulllabel)
ax2.set_xlabel('-log10(p-value) threshold in replication cohort', fontsize=10)
# ax2.set_title('Replication in UKBB', fontsize=12)
ax2.set_ybound(lower=-2, upper=42)
ax2.legend()

# fig.suptitle('Replication of autosomal Hellmann-Heimbach associations:\nNumber of SNPs replicating at different p-value thresholds', fontsize=15)
fig.suptitle('Replication of autosomal Hellmann-Heimbach associations', fontsize=12, y=0.95)


# Save and show the figure
fig.tight_layout()
plt.savefig("Replication_power_separate.pdf", bbox_inches='tight')
plt.show()

# # Plot the results
# f,ax=plt.subplots(figsize=(10, 6))
# plt.plot(MADCaP_thresholds, MADCaP_snp_counts, marker='s', color='green',linestyle='dotted', label=replabel_power+' in MADCaP')
# plt.plot(UKBB_thresholds, UKBB_snp_counts, marker='s', color='orange',linestyle='dotted', label=replabel_power+' in UKBB')
# plt.plot(MADCaP_thresholds, MADCaP_actual, marker='o', color='green', label=replabel_empirical+' in MADCaP')
# plt.plot(UKBB_thresholds, UKBB_actual, marker='o', color='orange', label=replabel_empirical+' in UKBB')
# plt.plot(UKBB_thresholds, Null, marker='s', color='blue',linestyle='dotted', label=nulllabel)
# plt.xlabel('-log10(p-value) threshold in replication cohort', fontsize=10)
# plt.ylabel('Number of SNPs', fontsize=10)
# ax.set_ybound(lower=-2, upper=42)
# # ax.set_ylim((0, 42))
# # ax.set_xlim((0.5, 5))
# # ax.autoscale_view()
# # plt.title('  Replication of autosomal Hellmann-Heimbach associations:\nNumber of SNPs replicating at different p-value thresholds', fontsize=15)
# plt.title('  Replication of autosomal Hellmann-Heimbach associations', fontsize=12)
# plt.legend()
# f.savefig("Replication_power.pdf", bbox_inches='tight')
# plt.show()
# plt.close()

