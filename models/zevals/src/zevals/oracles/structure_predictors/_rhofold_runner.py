"""Isolated-environment runner for batched RhoFold+ inference.

This module intentionally remains compatible with RhoFold+'s Python 3.7
environment. It is executed by path rather than imported into Zevals.
"""

from __future__ import print_function

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _read_fasta(path):
    records = []
    name = None
    parts = []
    with open(str(path)) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts).upper()))
                name = line[1:].split()[0]
                parts = []
            else:
                if name is None:
                    raise ValueError("Sequence data precedes the first FASTA header")
                parts.append(line)
    if name is not None:
        records.append((name, "".join(parts).upper()))
    return records


def _write_single_fasta(path, name, sequence):
    with open(str(path), "w") as handle:
        handle.write(">{}\n{}\n".format(name, sequence))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--input-fas", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--relax-steps", type=int, default=0)
    parser.add_argument("--single-sequence", action="store_true")
    parser.add_argument("--msa-manifest")
    parser.add_argument("--database-path")
    parser.add_argument("--binary-path")
    args = parser.parse_args()
    if args.single_sequence and args.msa_manifest:
        raise ValueError("single-sequence and msa-manifest are mutually exclusive")
    if not args.single_sequence and not args.msa_manifest:
        if not args.database_path or not args.binary_path:
            raise ValueError(
                "Generated-MSA mode requires database-path and binary-path"
            )

    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path))
    from rhofold.config import rhofold_config
    from rhofold.rhofold import RhoFold
    from rhofold.utils import get_device, save_ss2ct
    from rhofold.utils.alphabet import get_features

    msa_manifest = {}
    if args.msa_manifest:
        with open(args.msa_manifest) as handle:
            msa_manifest = json.load(handle)

    records = _read_fasta(Path(args.input_fas))
    if not records:
        raise ValueError("No FASTA records found in {}".format(args.input_fas))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("Zevals RhoFold+")

    device = get_device(args.device)
    logger.info("Loading RhoFold+ checkpoint once for %d sequence(s)", len(records))
    model = RhoFold(rhofold_config)
    state = torch.load(args.ckpt, map_location=torch.device("cpu"))["model"]
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)

    for index, (name, sequence) in enumerate(records):
        record_dir = output_dir / "{:04d}".format(index)
        record_dir.mkdir(parents=True, exist_ok=True)
        input_fasta = record_dir / "input.fasta"
        _write_single_fasta(input_fasta, name, sequence)

        if args.single_sequence:
            input_a3m = input_fasta
        elif msa_manifest:
            if name not in msa_manifest:
                raise KeyError("No supplied MSA for FASTA record {}".format(name))
            input_a3m = Path(msa_manifest[name])
        else:
            from rhofold.data.balstn import BLASTN

            input_a3m = record_dir / "input.a3m"
            databases = [
                str(Path(args.database_path) / "rnacentral.fasta"),
                str(Path(args.database_path) / "nt"),
            ]
            blast = BLASTN(binary_dpath=args.binary_path, databases=databases)
            blast.query(str(input_fasta), str(input_a3m), logger)

        data = get_features(str(input_fasta), str(input_a3m))
        outputs = model(
            tokens=data["tokens"].to(device),
            rna_fm_tokens=data["rna_fm_tokens"].to(device),
            seq=data["seq"],
        )
        output = outputs[-1]
        coordinates = output["cord_tns_pred"][-1].squeeze(0)
        plddt = output["plddt"][0]
        ss_prob_map = torch.sigmoid(output["ss"][0, 0]).detach().cpu().numpy()
        save_ss2ct(
            ss_prob_map,
            data["seq"],
            str(record_dir / "ss.ct"),
            threshold=0.5,
        )
        np.savez_compressed(
            str(record_dir / "results.npz"),
            dist_n=torch.softmax(output["n"].squeeze(0), dim=0).detach().cpu().numpy(),
            dist_p=torch.softmax(output["p"].squeeze(0), dim=0).detach().cpu().numpy(),
            dist_c=torch.softmax(output["c4_"].squeeze(0), dim=0)
            .detach()
            .cpu()
            .numpy(),
            ss_prob_map=ss_prob_map,
            plddt=plddt.detach().cpu().numpy(),
        )
        unrelaxed_path = record_dir / "unrelaxed_model.pdb"
        model.structure_module.converter.export_pdb_file(
            data["seq"],
            coordinates.detach().cpu().numpy(),
            path=str(unrelaxed_path),
            chain_id=None,
            confidence=plddt.detach().cpu().numpy(),
            logger=logger,
        )
        if args.relax_steps > 0:
            from rhofold.relax.relax import AmberRelaxation

            relaxed_path = record_dir / "relaxed_{}_model.pdb".format(args.relax_steps)
            relaxation = AmberRelaxation(max_iterations=args.relax_steps, logger=logger)
            relaxation.process(str(unrelaxed_path), str(relaxed_path))
        logger.info("Folded %s (%d nt)", name, len(sequence))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
