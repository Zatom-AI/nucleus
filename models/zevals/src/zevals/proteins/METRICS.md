# Evaluation Metrics for Protein Design

## 1. Common Backbone Metrics (both papers)

These metrics treat each generated backbone as a Cα-only structure and ask whether valid sequences can be found for it, whether the generated set is diverse, and whether it is structurally novel.

---

### 1.1 Designability (scRMSD / Self-Consistency RMSD)

**What it measures.** Whether there exists *any* amino acid sequence that folds into the generated backbone. This is a structural quality metric and does not rely on the generated sequence (if any).

**Protocol.**

1. Generate a set of protein backbones (Cα coordinates only).
2. For each backbone, use **ProteinMPNN** (Dauparas et al., 2022) to sample *M* candidate sequences conditioned on that backbone structure, at sampling temperature 0.1. Proteína uses *M* = 8 (written as PMPNN-8); La-Proteína reports both PMPNN-1 and PMPNN-8 variants.
3. Fold each candidate sequence into a 3D structure with **ESMFold** (Lin et al., Science 2023).
4. Compute the Cα RMSD (scRMSD) between the ESMFold-predicted structure and the original generated backbone after optimal superposition.
5. A sample is **designable** if the minimum scRMSD over the *M* sequences is < **2 Å**.
6. Report **Designability (%)** = fraction of generated backbones that are designable.

**Tools / frozen oracles.**
- ProteinMPNN (frozen inverse-folding model)
- ESMFold (frozen structure prediction LM)

**Implementation notes.** Both papers generate 100 proteins per length for lengths in a standard set (e.g., {50, 100, 150, 200, 250} residues for Proteína's main benchmark; {100, 200, 300, 400, 500} for La-Proteína's all-atom benchmark). Proteína also reports results separately for long chains up to 800 residues.

Zevals keeps the default designability definition scRMSD-only, matching the
published protocol above. A mean ESMFold pLDDT gate can be enabled explicitly
with `pipeline.plddt_threshold`.

---

### 1.2 Diversity

**What it measures.** Whether the generated set spans a wide range of distinct folds, rather than collapsing to a few repeated structures.

**Protocol.**

1. Take the set of *designable* backbone samples.
2. Run **Foldseek** `easy-cluster` with TM-score-based structural alignment (`--alignment-type 1`), using a TM-score threshold of **0.5** to define cluster membership (structures with TM-score > 0.5 are considered the same fold).
3. Diversity = **number of clusters / number of designable samples** (cluster ratio, in [0, 1]).

Proteína additionally reports **number of designable clusters** as a raw integer, and a length-averaged **mean pairwise TM-score** (lower = more diverse), but the primary diversity figure in both papers is the cluster ratio.

**Tools.**
- Foldseek (structural alignment and clustering)

**Notes.** Diversity is computed only over *designable* samples, so it is sensitive to the designability threshold. The Foldseek command used by Proteína (and cited by downstream papers):

```bash
foldseek easy-cluster input_dir clust tmp \
  --alignment-type 1 --cov-mode 0 \
  --min-seq-id 0 --tmscore-threshold 0.5
```

**TM-align complete-linkage AUC variant.** Some benchmarks instead compute a
full pairwise TM-align matrix over the generated backbones, symmetrize
TMalign's asymmetric TM scores with `min(TM_ij, TM_ji)`, cluster by complete
linkage so every pair in a cluster has TM above the swept threshold, and
integrate the cluster fraction over TM thresholds from 0 to 1. Higher AUC
means the generated set stays split into more clusters at lower thresholds,
i.e. it is more structurally diverse. In zevals this is exposed as
`compute_tmalign_complete_linkage_diversity` and can be run on either all
generated samples or the designable subset.

---

### 1.3 Novelty

**What it measures.** Whether the generated backbones differ meaningfully from known structures in the PDB and AFDB (i.e., the model is not simply memorizing training data).

**Protocol.**

1. Take the set of *designable* backbone samples.
2. For each generated structure, use **Foldseek** `easy-search` to find the *closest match* (highest TM-score hit) in the reference database (PDB or AFDB separately).
3. Novelty = average maximum TM-score across all designable samples.
4. **Lower is better** (more novel): TM-score near 0.5 suggests a distant homologue; near 1.0 means near-exact reproduction.

Proteína reports novelty vs. **PDB** and vs. **AFDB** as two separate columns.

**Tools.**
- Foldseek (structural search)

**Notes.** Foldseek command pattern:
```bash
foldseek easy-search input_dir pdb_db result.m8 tmp \
  --alignment-type 1 --tmscore-threshold 0.0
```
The metric is the mean of (1 − max TM-score) in some papers, but Proteína reports the raw average maximum TM-score directly (lower = more novel).

---

### 1.4 Secondary Structure Content (α-helix and β-sheet %)

**What it measures.** The fraction of generated residues assigned to α-helices versus β-strands. A known failure mode of protein backbone generators is over-production of α-helices and under-production of β-sheets.

**Protocol.**

1. For each generated backbone (Cα only), assign secondary structure using the **P-SEA algorithm** (Labesse et al., 1997), which is implemented in the **Biotite** library.
2. Compute the fraction of residues assigned to each secondary structure class (helix, strand, coil) across all generated samples.
3. Compare to reference distributions from the PDB or AFDB.

**Tools.**
- P-SEA (secondary structure assignment from Cα trace alone; Biotite library)

**Note.** Unlike DSSP, P-SEA operates on Cα coordinates only (no side chains needed), making it appropriate for backbone-only generated structures.

---

## 2. All-Atom Co-Designability (La-Proteína only)

La-Proteína generates sequences *jointly* with structures. Rather than just checking whether *some* sequence can fold into the backbone (designability), co-designability checks whether the *co-generated* sequence folds into the *co-generated* full-atom structure.

### 2.1 All-Atom Co-Designability

**What it measures.** Whether the jointly generated (sequence, structure) pair is internally consistent at atomic resolution.

**Protocol.**

1. La-Proteína generates a fully atomistic structure (all heavy atoms in Atom37 format) and a companion amino acid sequence together.
2. Fold the *generated sequence* with **ESMFold** to obtain a predicted structure.
3. Compute the **all-atom RMSD** (over all heavy atoms, not just Cα) between the generated atomistic structure and the ESMFold-predicted structure.
4. A sample is **all-atom co-designable** if all-atom RMSD < **2 Å**.
5. Report **Co-designability (%)** = fraction of samples passing this threshold.

**Tools.**
- ESMFold (frozen)

**Variants.** La-Proteína also reports a **Cα co-designability** (using only Cα RMSD < 2 Å), which is a coarser backbone-level version. The main headline metric is all-atom.

---

### 2.2 pLDDT (Confidence Score)

**What it measures.** The average ESMFold confidence over successfully co-designed samples.

**Protocol.**

1. Run ESMFold on the generated sequences.
2. ESMFold outputs per-residue pLDDT scores (0–100).
3. Average pLDDT across all residues for each successfully co-designable sample.
4. Average across all co-designable samples.

**Threshold.** pLDDT ≥ 80 is considered high-confidence (following AlphaFold2/ESMFold conventions).

**Tools.**
- ESMFold (frozen)

---

### 2.3 Co-Design Diversity

**What it measures.** Whether the set of co-designable structures covers a broad range of folds.

**Protocol.**

1. Take all all-atom co-designable samples.
2. Run Foldseek `easy-cluster` at TM-score threshold 0.5 (same as backbone diversity above).
3. Report number of clusters.

**Tools.**
- Foldseek

---

## 3. Biophysical / Structural Validity Metrics (La-Proteína only)

Because La-Proteína generates side chains, it is evaluated on physical realism — whether the atoms are placed in geometrically valid, sterically feasible arrangements.

### 3.1 Side-Chain Torsion Angle Distribution (Rotamer Analysis)

**What it measures.** Whether the model correctly learns the distribution of side-chain dihedral angles (χ angles) for each residue type, particularly that side chains adopt physically stable rotameric states.

**Protocol.**

1. For each generated protein, extract all side-chain torsion angles (χ1, χ2, … for each residue type) from the generated atomistic structure.
2. Compute a kernel density estimate (histogram) of the χ1 angle distribution for each amino acid type (e.g., TRP χ1).
3. Compare the generated distribution to reference distributions from the **PDB** and **AFDB** using visual overlay.
4. Evaluate rotamer population frequencies: well-folded structures should reproduce the known rotameric states (gauche+, gauche−, trans) and their relative frequencies as in crystallographic rotamer libraries (Lovell et al., 2000; Dunbrack library).

**Tools.**
- Custom analysis on Atom37-format coordinates (BioPython / OpenMM / in-house code)
- Reference PDB/AFDB rotamer distributions

La-Proteína reports χ1 plots for all 20 amino acid types in the appendix.

---

## 4. Summary Table

| Metric | Paper | Level | Oracle(s) / Tools | Threshold / Target | Direction |
|--------|-------|-------|-------------------|--------------------|-----------|
| Designability (PMPNN-8) | Both | Backbone | ProteinMPNN + ESMFold | Cα scRMSD < 2 Å | ↑ |
| Designability (PMPNN-1) | La-Proteína | Backbone | ProteinMPNN + ESMFold | Cα scRMSD < 2 Å | ↑ |
| Diversity (cluster ratio) | Both | Backbone | Foldseek (TM-score ≥ 0.5 = same cluster) | — | ↑ |
| Diversity (# clusters) | Both | Backbone | Foldseek | — | ↑ |
| Novelty vs. PDB | Both | Backbone | Foldseek search against PDB | Mean max TM-score | ↓ |
| Novelty vs. AFDB | Both | Backbone | Foldseek search against AFDB | Mean max TM-score | ↓ |
| Secondary structure % (α/β) | Both | Backbone | P-SEA (Biotite) | Compare to PDB/AFDB | — |
| All-atom Co-designability | La-Proteína | All-atom | ESMFold | All-atom RMSD < 2 Å | ↑ |
| Cα Co-designability | La-Proteína | Backbone | ESMFold | Cα RMSD < 2 Å | ↑ |
| pLDDT | La-Proteína | Confidence | ESMFold | ≥ 80 high-confidence | ↑ |

---

## 5. Key Implementation Notes

**Sample sizes.** Proteína evaluates distributional metrics (FPSD, fJSD, fS) on **5,000 samples**; backbone metrics (designability, diversity, novelty) use 100 samples per length. La-Proteína uses 100 samples per length for its main all-atom benchmark across lengths {100, 200, 300, 400, 500}.

**Noise scaling.** Both papers note that most baselines and model variants use *reduced noise / reduced temperature* during sampling (e.g., noise scale γ < 1 in Proteína; ηx, ηz < 1 in La-Proteína). This trades diversity for higher designability. Results at the "full distribution" (no noise reduction) are reported separately in Proteína's Table 3, and there the distributional metrics are significantly better, showing these metrics capture information that sample-level metrics miss.

**ESMFold version.** Both papers use ESMFold (Lin et al., Science 2023) as the single structure prediction oracle. Some concurrent papers validate with AlphaFold3 as an independent check.

**Secondary structure (Proteína).** Reported as percentage of residues assigned to α-helix and β-strand by P-SEA, averaged over the full generated set (not just designable samples). The α and β columns in the tables reflect this.

---

## 6. References

- Geffner et al. (2025). *Proteína: Scaling Flow-based Protein Structure Generative Models.* ICLR 2025. arXiv:2503.00710
- Geffner et al. (2025). *La-Proteína: Atomistic Protein Generation via Partially Latent Flow Matching.* arXiv:2507.09466
- Dauparas et al. (2022). *Robust deep learning–based protein sequence design using ProteinMPNN.* Science.
- Lin et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model (ESMFold).* Science.
- van Kempen et al. (2024). *Fast and accurate protein structure search with Foldseek.* Nature Biotechnology.
- Steinegger & Söding (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.* Nature Biotechnology.
- Labesse et al. (1997). *P-SEA: a new efficient assignment of secondary structure from Cα.* CABIOS.
- Dawson et al. (2016). *CATH: an expanded resource to predict protein function through structure and sequence.* Nucleic Acids Research.
- Lovell et al. (2000). *The penultimate rotamer library.* Proteins.
- Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium [FID].* NeurIPS.
- Salimans et al. (2016). *Improved Techniques for Training GANs [Inception Score].* NeurIPS.
