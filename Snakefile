import os
import pandas as pd


configfile: "files/config.yaml"


OUTDIR = config["output"]["dir"]


# assumed unique row identify linking to embryo name
embryo_log = pd.read_csv(config["input"]["logfile"], index_col=0)
EMBRYOS = glob_wildcards(
    os.path.join(
        config['input']['datadir'], "{embryo}.nd2"
    )
).embryo

rule all:
    input:
        os.path.join(OUTDIR, "final", "counts.csv"),


def get_embryo_param(wc, col):
    return embryo_log.at[wc.embryo, col]


rule normalize_pmc_stains:
    input:
        image=os.path.join(config['input']['datadir'], "{embryo}.nd2"),
    params:
        channel_name="pmc",
        channels=lambda wc: get_embryo_param(wc, "channel_order"),
        z_start=lambda wc: get_embryo_param(wc, "z-start"),
        z_end=lambda wc: get_embryo_param(wc, "z-end"),
    output:
        h5=temp(os.path.join(OUTDIR, "pmc_norm", "{embryo}.h5"),)
    conda:
        "envs/hcr_quant.yaml"
    script:
        "scripts/normalize_pmc_stain.py"


rule predict_pmcs:
    input:
        image=os.path.join(OUTDIR, "pmc_norm", "{embryo}.h5"),
        model=config["ilastik"]["model"],
    params:
        ilastik_loc=config["ilastik"]["loc"],
    output:
        temp(os.path.join(OUTDIR, "pmc_probs", "{embryo}.h5"))
    log:
        os.path.join(OUTDIR, "logs", "prediction", "{embryo}.log"),
    shell:
        "cd $(dirname {params.ilastik_loc}); "
        "(./run_ilastik.sh --headless "
        "--project={input.model} "
        "--output_format=hdf5 "
        "--output_filename_format={output} "
        "{input.image}) 2> {log}"


rule label_pmcs:
    input:
        stain=os.path.join(OUTDIR, "pmc_norm", "{embryo}.h5"),
        probs=os.path.join(OUTDIR, "pmc_probs", "{embryo}.h5"),
    output:
        labels=os.path.join(OUTDIR, "labels", "{embryo}_pmc_labels.h5"),
    log:
        log=os.path.join("logs", "labels", "{embryo}.log"),
    conda:
        "envs/segmentation.yaml"
    script:
        "scripts/label_pmcs.py"


rule quantify_expression:
    input:
        image=os.path.join(config['input']['datadir'], "{embryo}.nd2"),
        labels=os.path.join(OUTDIR, "labels", "{embryo}_pmc_labels.h5"),
    params:
        gene_params=config["quant"]["genes"],
        channels=lambda wc: get_embryo_param(wc, "channel_order"),
        z_start=lambda wc: get_embryo_param(wc, "z-start"),
        z_end=lambda wc: get_embryo_param(wc, "z-end"),
    output:
        image=os.path.join(OUTDIR, "expression", "{embryo}.nc"),
        csv=os.path.join(OUTDIR, "counts", "{embryo}.csv"),
    log:
        "logs/quant/{embryo}.log",
    conda:
        "envs/hcr_quant.yaml"
    script:
        "scripts/count_spots.py"


rule combine_counts:
    input:
        expand(
            os.path.join(OUTDIR, "counts", "{embryo}.csv"),
            embryo=EMBRYOS
        )
    output:
        os.path.join(OUTDIR, "final", "counts.csv"),
    script:
        "scripts/combine_counts.py"
