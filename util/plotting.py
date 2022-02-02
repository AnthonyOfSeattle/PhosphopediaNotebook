import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fdr(build):
    cutoff_label = ["PSM", "Peptide", "Site"]
    cutoff_score = [build.psms[build.psms.qvalue < 0.01].score.min(),
                    build.peptides[build.peptides.qvalue < 0.01].score.min(),
                    build.sites[build.sites.qvalue < 0.01].score.min()]
    
    level = []
    detection_set = []
    n_targets = []
    n_detections = []
    for l, s in zip(cutoff_label, cutoff_score):
        level.extend([l]*3)
        detection_set.extend(["PSM", "Peptide", "Site"])
        n_targets.extend([np.sum(build.psms[build.psms.score >= s].label == "target"),
                          np.sum(build.peptides[build.peptides.score >= s].label == "target"),
                          np.sum(build.sites[build.sites.score >= s].label == "target")])
        n_detections.extend([np.sum(build.psms.score >= s),
                             np.sum(build.peptides.score >= s),
                             np.sum(build.sites.score >= s)])
        
    fdr_df = pd.DataFrame({"level": level,
                           "set": detection_set,
                           "targets": n_targets,
                           "detections" : n_detections})
    fdr_df["fdr"] = (fdr_df.detections - fdr_df.targets)/fdr_df.targets
    print(fdr_df)
    
    plt.figure(figsize=(8, 8))

    plt.axhline(0.01, linestyle="--", lw=3,
                color="black", alpha=.5, 
                zorder=0)

    sns.barplot(x="level", y="fdr", hue="set", data = fdr_df,
                palette=["#461554", "#56c566", "#30708d"])

    plt.tick_params(labelsize=32)
    plt.xlabel("FDR Filter Level", size=32)
    plt.ylabel("FDR", size=32)
    plt.legend(title="", fontsize=28)
    plt.yscale("log")
    

def plot_detections(build, 
                    compare):
    
    fig = plt.figure(figsize = [20, 7], constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    
    psm_filter = build.psms[build.psms.qvalue < 0.01].score.min()
    pep_filter = build.peptides[build.peptides.qvalue < 0.01].score.min()
    site_filter = build.sites[build.sites.qvalue < 0.01].score.min()
    
    # PSM analysis
    ax1 = fig.add_subplot(gs[:, 0])
    for lab, v in zip(["No Filter", "1% PSM", "1% Peptide", "1% Site"],
                  [np.sum(build.psms.label == "target"),
                   np.sum(build.psms[build.psms.score > psm_filter].label == "target"),
                   np.sum(build.psms[build.psms.score > pep_filter].label == "target"),
                   np.sum(build.psms[build.psms.score > site_filter].label == "target")]):
        plt.bar(lab, v, color="black", alpha=.5)
        
    ax1.bar("Original", compare["psms"], color = "blue", alpha=.5)
    ax1.tick_params(labelsize=20)
    plt.xticks(rotation=15, ha="right")
    
    # Peptide analysis
    ax2 = fig.add_subplot(gs[0, 1])
    for lab, v in zip(["1% Peptide", "1% Site"],
                  [np.sum(build.peptides[build.peptides.score > pep_filter].label == "target"),
                   np.sum(build.peptides[build.peptides.score > site_filter].label == "target")]):
        ax2.bar(lab, v, color="black", alpha=.5)

    ax2.bar("Original", compare["peptides"], color = "blue", alpha=.5)
    ax2.tick_params(labelsize=20)
    plt.xticks(rotation=15, ha="right")
    
    # Site analysis
    ax3 = fig.add_subplot(gs[1, 1])
    for lab, v in zip(["1% Site"],
                  [np.sum(build.sites[build.sites.score > site_filter].label == "target")]):
        ax3.bar(lab, v, color="black", alpha=.5)
        
    ax3.bar("Original", compare["sites"], color = "blue", alpha=.5)
    ax3.tick_params(labelsize=20)
    plt.xticks(rotation=15, ha="right")

    
def plot_filewise_fdr(build):
    filtered_psms = build.psms[build.psms.qvalue < 0.01]
    
    td_counts = filtered_psms.groupby(["sample_name", "label"])\
                             .id\
                             .count()\
                             .rename("n")\
                             .reset_index()
    td_counts = td_counts.pivot(index="sample_name", columns="label", values="n").fillna(0.)
    td_counts["fdr"] = (td_counts["decoy"] + 1)/td_counts["target"]
    
    plt.figure(figsize=(10, 8))

    cutoff_select = td_counts.fdr < 0.02
    sns.scatterplot(data = td_counts[cutoff_select],
                    x="target", y="fdr",
                    color="#30708d", 
                    linewidth=0,
                    alpha=.25)
    
    sns.scatterplot(data = td_counts[~cutoff_select],
                    x="target", y="fdr",
                    color="#461554", 
                    linewidth=0,
                    alpha=.25)

    plt.axhline(.01, c="black", linestyle="--", lw=3,  zorder=0)
    plt.axhline(.02, c="black", linestyle="--", lw=3, alpha=.5, zorder=0)
    plt.text(3e4, 0.012, "Passing: {}".format(np.sum(cutoff_select)), ha="right", size=28)
    plt.text(3e4, 0.025, "Failing: {}".format(np.sum(~cutoff_select)), ha="right", size=28)

    plt.tick_params(labelsize=32)

    plt.xlabel("Contributed PSMs", size=32)
    plt.ylabel("FDR in Log10 scale", size=32)
    plt.yscale("log")
    

def plot_number_of_charges(build):
    psm_charges = build.psms[build.psms.qvalue < 0.01]\
                       .loc[:, ["pep_id", "precursor_charge"]]
    peptides = build.peptides[build.peptides.qvalue < 0.01]\
                    .loc[:, ["id"]]
    
    peptide_charge = peptides.join(psm_charges.set_index("pep_id"), on="id")
    peptide_charge = peptide_charge[~peptide_charge.precursor_charge.isna()]
    peptide_charge.precursor_charge = peptide_charge.precursor_charge.astype(int)
    
    n_distinct_states = peptide_charge.groupby("id").precursor_charge\
                                      .apply(lambda s: s.unique().shape[0])\
                                      .rename("n_charge_states")
    
    states, counts = np.unique(n_distinct_states, return_counts=True)
    
    fig = plt.figure(figsize = (5, 5))
    
    plt.bar(states, 100*counts/np.sum(counts),
            color="#461554",
            alpha=.75)
    
    plt.ylim(0, 60)
    plt.xticks(states)
    plt.xlabel("Number of\nCharge States", size=32)
    plt.ylabel("Percent of\nPeptides", size=32)
    plt.tick_params(labelsize=32)
    
    fig.get_axes()[0].spines["right"].set_visible(False)
    fig.get_axes()[0].spines["top"].set_visible(False)

    
def plot_number_of_analyzers(build, analyzers):
    psm_analyzers = build.psms[build.psms.qvalue < 0.01]\
                         .loc[:, ["pep_id", "sample_name"]]\
                         .join(analyzers.set_index("sampleName"),
                               on="sample_name")

    peptides = build.peptides[build.peptides.qvalue < 0.01]\
                    .loc[:, ["id"]]
    
    peptide_analyzers = peptides.join(psm_analyzers.set_index("pep_id"), on="id")
    
    peptide_class  = peptide_analyzers.sort_values(["id", "ms2Analyzer"])\
                                      .groupby("id").ms2Analyzer\
                                      .apply(lambda s: "+".join(s.unique()))\
                                      .rename("class")
    peptide_class[peptide_class == "FTMS+ITMS"] = "Both"
    
    states, counts = np.unique(peptide_class, return_counts=True)
    
    fig = plt.figure(figsize = (5, 5))
    
    plt.bar(states, 100*counts/np.sum(counts),
            color="#461554",
            alpha=.75)
    
    plt.ylim(0, 60)
    plt.xticks(states)
    plt.yticks([0, 10, 20, 30, 40, 50, 60])
    plt.xlabel("", size=32)
    plt.ylabel("Percent of\nPeptides", size=32)
    plt.tick_params(labelsize=32)
    
    fig.get_axes()[0].spines["right"].set_visible(False)
    fig.get_axes()[0].spines["top"].set_visible(False)
