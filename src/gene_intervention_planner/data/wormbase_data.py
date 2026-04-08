from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
import requests

DEFAULT_RELEASE = "WBPS19"
DEFAULT_SPECIES = "caenorhabditis_elegans"
DEFAULT_PROJECT = "PRJNA13758"

DEFAULT_RAW_DIR = Path("data/raw/wormbase")
DEFAULT_PROCESSED_DIR = Path("data/processed")

DEFAULT_DOWNLOAD_SUFFIXES = (
    "genomic.fa",
    "annotations.gff3",
    "protein.fa",
    "cds.fa",
    "transcripts.fa",
    "orthologs.tsv",
    "phenotypes.gaf",
    "orthology_inferred_phenotypes.gaf",
)

EVIDENCE_WEIGHTS = {
    "IDA": 1.0,
    "IMP": 0.95,
    "IGI": 0.90,
    "IPI": 0.85,
    "IEP": 0.80,
    "TAS": 0.70,
    "NAS": 0.45,
    "IEA": 0.35,
}


def _release_base_url(*, release: str, species: str, project: str) -> str:
    return (
        "https://ftp.ebi.ac.uk/pub/databases/wormbase/parasite/releases/"
        f"{release}/species/{species}/{project}/"
    )


def _release_filename(*, species: str, project: str, release: str, suffix: str) -> str:
    return f"{species}.{project}.{release}.{suffix}.gz"


def _download_file(
    url: str, out_path: Path, *, force: bool, timeout_seconds: int
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return out_path

    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return out_path


def gunzip_file(path: Path, *, force: bool = False) -> Path:
    if path.suffix.lower() != ".gz":
        return path

    out_path = path.with_suffix("")
    if out_path.exists() and not force:
        return out_path

    with gzip.open(path, "rb") as source, out_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return out_path


def download_wormbase_release(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    release: str = DEFAULT_RELEASE,
    species: str = DEFAULT_SPECIES,
    project: str = DEFAULT_PROJECT,
    file_suffixes: Iterable[str] = DEFAULT_DOWNLOAD_SUFFIXES,
    force: bool = False,
    timeout_seconds: int = 180,
) -> list[Path]:
    """Download selected compressed files for a WormBase Parasite release."""
    raw_path = Path(raw_dir)
    base_url = _release_base_url(release=release, species=species, project=project)
    downloaded: list[Path] = []
    for suffix in file_suffixes:
        filename = _release_filename(
            species=species,
            project=project,
            release=release,
            suffix=suffix,
        )
        url = base_url + filename
        target = raw_path / filename
        downloaded.append(
            _download_file(url, target, force=force, timeout_seconds=timeout_seconds)
        )
    return downloaded


def _attributes_map(attributes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in attributes.split(";"):
        if not entry:
            continue
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


def _gene_id_from_attributes(attributes: dict[str, str]) -> str:
    gene_id = attributes.get("Name") or attributes.get("ID", "")
    if gene_id.startswith("Gene:"):
        gene_id = gene_id.split(":", 1)[1]
    if not gene_id and attributes.get("curie", "").startswith("WB:"):
        gene_id = attributes["curie"].split(":", 1)[1]
    return gene_id


def _gene_symbol_from_attributes(attributes: dict[str, str], gene_id: str) -> str:
    symbol = attributes.get("locus", "")
    if symbol:
        return symbol

    aliases = attributes.get("Alias", "")
    if aliases:
        return aliases.split(",", 1)[0]

    sequence_name = attributes.get("sequence_name", "")
    if sequence_name:
        return sequence_name
    return gene_id


def parse_wormbase_genes(gff3_path: str | Path) -> pl.DataFrame:
    """Extract gene-level records from WormBase GFF3."""
    rows: list[dict[str, str | int]] = []
    with Path(gff3_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            if fields[2] != "gene":
                continue

            attributes = _attributes_map(fields[8])
            gene_id = _gene_id_from_attributes(attributes)
            if not gene_id:
                continue

            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                continue

            rows.append(
                {
                    "gene_id": gene_id,
                    "gene_symbol": _gene_symbol_from_attributes(attributes, gene_id),
                    "chromosome": fields[0],
                    "start": start,
                    "end": end,
                    "strand": fields[6],
                    "biotype": attributes.get("biotype", "unknown"),
                    "gene_length": max(end - start + 1, 0),
                }
            )

    if not rows:
        raise ValueError(f"No gene features were found in {gff3_path}")

    return (
        pl.DataFrame(rows)
        .unique(subset=["gene_id"], keep="first", maintain_order=True)
        .sort("gene_id")
    )


def parse_wormbase_phenotypes(gaf_path: str | Path) -> pl.DataFrame:
    """Load and normalize WormBase phenotype annotations from GAF."""
    columns = [
        "db",
        "gene_id",
        "gene_symbol",
        "qualifier",
        "phenotype_id",
        "reference",
        "evidence_code",
        "aspect",
        "db_object_type",
        "taxon",
        "release_date",
        "assigned_by",
        "annotation_extension",
    ]
    df = pl.read_csv(
        gaf_path,
        separator="\t",
        has_header=False,
        comment_prefix="!",
        new_columns=columns,
        null_values=["", "."],
        infer_schema_length=5000,
    )
    if df.height == 0:
        raise ValueError(f"No phenotype annotations were found in {gaf_path}")

    normalized = (
        df.select(
            [
                pl.col("gene_id").cast(pl.Utf8, strict=False),
                pl.col("gene_symbol").cast(pl.Utf8, strict=False),
                pl.col("qualifier").cast(pl.Utf8, strict=False).fill_null(""),
                pl.col("phenotype_id").cast(pl.Utf8, strict=False),
                pl.col("reference").cast(pl.Utf8, strict=False).fill_null("unknown"),
                pl.col("evidence_code").cast(pl.Utf8, strict=False).fill_null("UNK"),
                pl.col("annotation_extension").cast(pl.Utf8, strict=False),
            ]
        )
        .with_columns(
            [
                pl.col("annotation_extension")
                .str.extract(r"^[^:]+:(.+)$", group_index=1)
                .fill_null(pl.col("phenotype_id"))
                .alias("phenotype_name"),
                pl.col("qualifier").str.contains("NOT").alias("is_not_annotation"),
                pl.col("evidence_code")
                .replace_strict(EVIDENCE_WEIGHTS, default=0.55)
                .cast(pl.Float64)
                .alias("feature_evidence_weight"),
            ]
        )
        .with_columns((~pl.col("is_not_annotation")).cast(pl.Int64).alias("label"))
        .unique(
            subset=[
                "gene_id",
                "phenotype_id",
                "reference",
                "qualifier",
                "evidence_code",
            ],
            keep="first",
            maintain_order=True,
        )
    )
    return normalized


def summarize_orthologs(orthologs_path: str | Path) -> pl.DataFrame:
    """Aggregate orthology features per gene_id."""
    path = Path(orthologs_path)
    if not path.exists():
        return pl.DataFrame(
            schema={
                "gene_id": pl.Utf8,
                "feature_ortholog_count": pl.Float64,
                "feature_human_ortholog_count": pl.Float64,
                "feature_max_query_identity": pl.Float64,
            }
        )

    return (
        pl.scan_csv(path, separator="\t", has_header=True, infer_schema_length=2000)
        .group_by("gene_id")
        .agg(
            [
                pl.len().cast(pl.Float64).alias("feature_ortholog_count"),
                pl.col("ortholog_species_name")
                .str.contains("Homo sapiens")
                .sum()
                .cast(pl.Float64)
                .alias("feature_human_ortholog_count"),
                pl.col("query_identity")
                .cast(pl.Float64, strict=False)
                .max()
                .fill_null(0.0)
                .alias("feature_max_query_identity"),
            ]
        )
        .collect()
    )


def build_gene_summary(
    *,
    genes_df: pl.DataFrame,
    phenotypes_df: pl.DataFrame,
    orthologs_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build a gene-level merged feature table from WormBase processed inputs."""
    phenotype_summary = phenotypes_df.group_by("gene_id").agg(
        [
            pl.first("gene_symbol").alias("gene_symbol_from_phenotype"),
            pl.len().cast(pl.Float64).alias("annotation_count"),
            pl.col("label").sum().cast(pl.Float64).alias("positive_annotation_count"),
            (pl.len() - pl.col("label").sum())
            .cast(pl.Float64)
            .alias("negative_annotation_count"),
            pl.n_unique("phenotype_id")
            .cast(pl.Float64)
            .alias("unique_phenotype_count"),
            pl.n_unique("reference").cast(pl.Float64).alias("reference_count"),
            pl.col("feature_evidence_weight")
            .mean()
            .cast(pl.Float64)
            .alias("mean_evidence_weight"),
        ]
    )

    merged = genes_df.join(phenotype_summary, on="gene_id", how="left")
    if orthologs_df is not None and orthologs_df.height > 0:
        merged = merged.join(orthologs_df, on="gene_id", how="left")

    merged = merged.with_columns(
        [
            pl.coalesce(["gene_symbol", "gene_symbol_from_phenotype", "gene_id"]).alias(
                "gene_symbol"
            ),
            pl.col("annotation_count").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("positive_annotation_count")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("negative_annotation_count")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("unique_phenotype_count")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("reference_count").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("mean_evidence_weight")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("feature_ortholog_count")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("feature_human_ortholog_count")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
            pl.col("feature_max_query_identity")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0),
        ]
    ).with_columns(
        [
            (
                pl.when(pl.col("annotation_count") > 0)
                .then(pl.col("positive_annotation_count") / pl.col("annotation_count"))
                .otherwise(0.0)
            ).alias("positive_annotation_rate"),
            (
                pl.when(pl.col("gene_length") > 0)
                .then(1000.0 * pl.col("annotation_count") / pl.col("gene_length"))
                .otherwise(0.0)
            ).alias("annotation_density_per_kb"),
        ]
    )

    return (
        merged.drop("gene_symbol_from_phenotype")
        .sort(["annotation_count", "gene_id"], descending=[True, False])
        .with_row_index("gene_rank", offset=1)
    )


def preprocess_wormbase_release(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    release: str = DEFAULT_RELEASE,
    species: str = DEFAULT_SPECIES,
    project: str = DEFAULT_PROJECT,
    focus_genes: Sequence[str] = ("gap-2",),
    comparator_genes: Sequence[str] = ("nlg-1", "nrx-1"),
    max_context_genes: int = 140,
) -> dict[str, Path]:
    """Create processed tables from downloaded WormBase files."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{species}.{project}.{release}"
    gff3_path = raw_path / f"{prefix}.annotations.gff3"
    gaf_path = raw_path / f"{prefix}.phenotypes.gaf"
    orthologs_path = raw_path / f"{prefix}.orthologs.tsv"

    for required in (gff3_path, gaf_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Required WormBase file not found: {required}. "
                "Download release files first or provide an existing raw directory."
            )

    genes_df = parse_wormbase_genes(gff3_path)
    phenotype_df = parse_wormbase_phenotypes(gaf_path)
    ortholog_df = summarize_orthologs(orthologs_path)
    merged_df = build_gene_summary(
        genes_df=genes_df,
        phenotypes_df=phenotype_df,
        orthologs_df=ortholog_df,
    )

    genes_out = processed_path / "wormbase_genes.parquet"
    phenotypes_out = processed_path / "wormbase_gene_phenotypes.parquet"
    merged_parquet_out = processed_path / "wormbase_merged.parquet"
    merged_csv_out = processed_path / "wormbase_merged.csv"
    ortholog_summary_out = processed_path / "wormbase_ortholog_summary.parquet"

    genes_df.write_parquet(genes_out)
    phenotype_df.write_parquet(phenotypes_out)
    merged_df.write_parquet(merged_parquet_out)
    merged_df.write_csv(merged_csv_out)
    ortholog_df.write_parquet(ortholog_summary_out)

    candidate_output = processed_path / "wormbase_gap2_candidates.parquet"
    candidate_parquet, candidate_csv = build_and_save_gap2_candidates(
        phenotypes_gaf_path=gaf_path,
        output_path=candidate_output,
        focus_genes=focus_genes,
        comparator_genes=comparator_genes,
        orthologs_tsv_path=orthologs_path if orthologs_path.exists() else None,
        max_context_genes=max_context_genes,
    )

    return {
        "genes_parquet": genes_out,
        "gene_phenotypes_parquet": phenotypes_out,
        "ortholog_summary_parquet": ortholog_summary_out,
        "merged_parquet": merged_parquet_out,
        "merged_csv": merged_csv_out,
        "gap2_candidates_parquet": candidate_parquet,
        "gap2_candidates_csv": candidate_csv,
    }


def download_and_preprocess_wormbase(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    release: str = DEFAULT_RELEASE,
    species: str = DEFAULT_SPECIES,
    project: str = DEFAULT_PROJECT,
    download: bool = True,
    force_download: bool = False,
    force_decompress: bool = False,
    focus_genes: Sequence[str] = ("gap-2",),
    comparator_genes: Sequence[str] = ("nlg-1", "nrx-1"),
    max_context_genes: int = 140,
) -> dict[str, Path]:
    """End-to-end real WormBase data preparation into `data/processed`."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    if download:
        downloaded = download_wormbase_release(
            raw_dir=raw_path,
            release=release,
            species=species,
            project=project,
            force=force_download,
        )
        for path in downloaded:
            gunzip_file(path, force=force_decompress)

    return preprocess_wormbase_release(
        raw_dir=raw_path,
        processed_dir=processed_dir,
        release=release,
        species=species,
        project=project,
        focus_genes=focus_genes,
        comparator_genes=comparator_genes,
        max_context_genes=max_context_genes,
    )
