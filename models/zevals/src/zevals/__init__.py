"""zevals — self-consistency and structure validation for Zatom generative models.

Provides post-hoc evaluation pipelines for five domains:
  - molecules: unconditional small-molecule generation
  - ligands:   protein-conditioned ligand design (PoseBusters, RMSD)
  - proteins:  protein backbone / sequence design (designability, novelty, diversity)
  - materials: crystal and periodic material generation (LeMatGenBench)
  - rna:       RNA backbone generation (validity, novelty, diversity)

Oracle backends (ESMFold, ProteinMPNN, PoseBusters) are swappable via
Hydra config groups under zevals/configs/oracle/.
"""

__version__ = "0.1.0"
