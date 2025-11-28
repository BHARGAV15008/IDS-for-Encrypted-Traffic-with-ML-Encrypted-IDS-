#!/usr/bin/env python3
"""
P22 Terminal Interface (Interactive)

Interactive, terminal-only workflow to run CSV/PCAP preprocessing, model demos,
adversarial training step, and ensemble evaluation without any UI.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich import print as rprint
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.data_ingestion.csv_loader import load_csv_dataset
from services.data_ingestion.pcap_loader import pcap_to_flows_with_cicflowmeter
from services.model_service.hybrid_attention_model import HybridCNNBiLSTMAttention
from services.adversarial_service.trainer import adversarial_training_step
from services.ensemble_service.ensemble_classifier import EnsembleClassifier
from pipeline.arff_pipeline import combine_arff_to_dataframe, train_evaluate_ensemble
from pipeline.deep_arff_pipeline import run_arff_deep
from pipeline.arff_analysis import analyze_arff_dirs, write_report
from pipeline.two_stage_arff_pipeline import run_two_stage

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _header() -> None:

    rprint(Panel.fit(
        "[bold cyan]P22 Encrypted Traffic IDS[/bold cyan]\nTerminal Interface",
        border_style="cyan"
    ))


@click.group()
def p22() -> None:

    _header()


@p22.command()
@click.option('--mode', type=click.Choice(['csv', 'pcap']), help='Dataset mode')
@click.option('--path', 'path_input', type=click.Path(), help='Path to CSV or PCAP')
def preprocess(mode: Optional[str], path_input: Optional[str]) -> None:

    mode = mode or Prompt.ask("Select dataset mode", choices=["csv", "pcap"], default="csv")
    path_input = path_input or Prompt.ask("Enter file path (.csv or .pcap)")
    path = Path(path_input)
    if not path.exists():
        rprint(f"[red]Path not found:[/red] {path}")
        sys.exit(1)

    if mode == 'csv':
        df = load_csv_dataset(str(path))
        rprint(f"[green]Loaded CSV:[/green] rows={len(df)}, cols={len(df.columns)}")
    else:
        out = Path(ROOT / '01_Data/02_Processed/flows_from_pcap.csv')
        df = pcap_to_flows_with_cicflowmeter(str(path), output_csv=str(out))
        rprint(f"[green]PCAP -> flows complete:[/green] saved={out} rows={len(df)}")


@p22.command()
@click.option('--seq-len', type=int, default=100, help='Sequence length for demo')
@click.option('--feat-dim', type=int, default=32, help='Feature dimension for demo')
@click.option('--num-classes', type=int, default=5, help='Number of classes')
def model_demo(seq_len: int, feat_dim: int, num_classes: int) -> None:

    batch = 8
    x = torch.randn(batch, seq_len, feat_dim)
    model = HybridCNNBiLSTMAttention(input_features=feat_dim, num_classes=num_classes)
    logits = model(x)
    rprint(f"[green]Model forward OK[/green] -> logits shape: {tuple(logits.shape)}")


@p22.command()
@click.option('--seq-len', type=int, default=100)
@click.option('--feat-dim', type=int, default=32)
@click.option('--num-classes', type=int, default=5)
@click.option('--epsilon', type=float, default=0.01)
def adversarial_step(seq_len: int, feat_dim: int, num_classes: int, epsilon: float) -> None:

    x = torch.randn(16, seq_len, feat_dim)
    y = torch.randint(0, num_classes, (16,))
    model = HybridCNNBiLSTMAttention(input_features=feat_dim, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = adversarial_training_step(model, loss_fn, optimizer, x, y, epsilon=epsilon)
    rprint(f"[green]Adversarial step OK[/green] -> loss: {loss:.4f}")


@p22.command()
@click.option('--feat-dim', type=int, default=128)
@click.option('--num-classes', type=int, default=5)
def ensemble_demo(feat_dim: int, num_classes: int) -> None:

    X = np.random.randn(200, feat_dim)
    y = np.random.randint(0, num_classes, size=200)
    clf = EnsembleClassifier(base_dim=feat_dim, num_classes=num_classes)
    clf.fit(X, y)
    preds = clf.predict(X[:10])
    rprint(f"[green]Ensemble OK[/green] -> preds: {preds.tolist()}")


@p22.command()
def wizard() -> None:

    mode = Prompt.ask("Dataset mode", choices=["csv", "pcap"], default="csv")
    file_path = Prompt.ask("Path to file")
    action = Prompt.ask("Action", choices=["preprocess", "model_demo", "adversarial_step", "ensemble_demo"], default="preprocess")

    if action == 'preprocess':
        preprocess.callback(mode, file_path)  # type: ignore
        return
    if action == 'model_demo':
        seq_len = IntPrompt.ask("seq_len", default=100)
        feat_dim = IntPrompt.ask("feat_dim", default=32)
        num_classes = IntPrompt.ask("num_classes", default=5)
        model_demo.callback(seq_len, feat_dim, num_classes)  # type: ignore
        return
    if action == 'adversarial_step':
        seq_len = IntPrompt.ask("seq_len", default=100)
        feat_dim = IntPrompt.ask("feat_dim", default=32)
        num_classes = IntPrompt.ask("num_classes", default=5)
        epsilon = click.prompt("epsilon", default=0.01, type=float)
        adversarial_step.callback(seq_len, feat_dim, num_classes, epsilon)  # type: ignore
        return
    if action == 'ensemble_demo':
        feat_dim = IntPrompt.ask("feat_dim", default=128)
        num_classes = IntPrompt.ask("num_classes", default=5)
        ensemble_demo.callback(feat_dim, num_classes)  # type: ignore
        return


@p22.command()
@click.option('--a1', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A1-ARFF'))
@click.option('--a2', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A2-ARFF'))
@click.option('--b', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario B-ARFF'))
def arff_combine_train(a1: str, a2: str, b: str) -> None:

    rprint("[cyan]Combining ARFF files and training ensemble...[/cyan]")
    df = combine_arff_to_dataframe([a1, a2, b])
    stats = train_evaluate_ensemble(df)
    rprint(Panel.fit(
        f"Samples: {stats['num_samples']}\n"
        f"Features: {stats['num_features']}\n"
        f"Classes: {stats['num_classes']}\n"
        f"Accuracy: {stats['accuracy']:.4f}\n"
        f"F1-weighted: {stats['f1_weighted']:.4f}\n\n"
        f"Report:\n{stats['report_text']}",
        title="ARFF Combined Ensemble Results",
        border_style="green"
    ))


@p22.command()
@click.option('--a1', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A1-ARFF'))
@click.option('--a2', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A2-ARFF'))
@click.option('--b', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario B-ARFF'))
@click.option('--seq-len', type=int, default=32)
@click.option('--batch-size', type=int, default=128)
@click.option('--epochs', type=int, default=15)
@click.option('--device', type=str, default='cpu')
@click.option('--no-focal', is_flag=True, help='Disable focal loss')
@click.option('--gamma', type=float, default=2.0, help='Focal loss gamma')
def arff_deep_train(a1: str, a2: str, b: str, seq_len: int, batch_size: int, epochs: int, device: str, no_focal: bool, gamma: float) -> None:

    rprint("[cyan]Training CNN+BiLSTM+Attention on combined ARFF...[/cyan]")
    metrics = run_arff_deep([a1, a2, b], seq_len=seq_len, batch_size=batch_size, epochs=epochs, device=device, use_focal=(not no_focal), gamma=gamma)
    rprint(Panel.fit(
        f"SeqLen: {metrics['seq_len']}  FeatDim: {metrics['feat_dim']}  Classes: {metrics['num_classes']}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"F1-weighted: {metrics['f1_weighted']:.4f}\n\n"
        f"Report:\n{metrics['report_text']}",
        title="ARFF Deep Model Results",
        border_style="green"
    ))


@p22.command()
@click.option('--a1', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A1-ARFF'))
@click.option('--a2', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A2-ARFF'))
@click.option('--b', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario B-ARFF'))
@click.option('--out', type=click.Path(), default=str(ROOT / '05_Evaluation/analysis_arff_report.txt'))
def arff_analyze(a1: str, a2: str, b: str, out: str) -> None:

    rprint("[cyan]Analyzing ARFF datasets (A1, A2, B)...[/cyan]")
    report = analyze_arff_dirs([a1, a2, b])
    write_report(report, out)
    rprint(Panel.fit(
        f"Common features: {report['summary']['common_features_count']}  Union features: {report['summary']['union_features_count']}\n"
        f"Common labels: {report['summary']['common_labels_count']}  Union labels: {report['summary']['union_labels_count']}\n\n"
        f"Report saved: {out}",
        title="ARFF Analysis",
        border_style="green"
    ))


@p22.command()
@click.option('--a1', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A1-ARFF'))
@click.option('--a2', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario A2-ARFF'))
@click.option('--b', type=click.Path(exists=True), default=str(ROOT / '01_Data/Scenario B-ARFF'))
def arff_two_stage_train(a1: str, a2: str, b: str) -> None:

    rprint("[cyan]Training two-stage pipeline: (1) Encryption detection, (2) Encrypted intrusion.[/cyan]")
    metrics = run_two_stage([a1, a2, b])
    rprint(Panel.fit(
        f"Stage1 (Encryption) -> Acc: {metrics['stage1_encryption_metrics']['accuracy']:.4f}  F1w: {metrics['stage1_encryption_metrics']['f1_weighted']:.4f}\n\n"
        f"Report1:\n{metrics['stage1_encryption_metrics']['report_text']}\n\n"
        f"Stage2 (Encrypted Intrusion) -> Acc: {metrics['stage2_encrypted_intrusion_metrics']['accuracy']:.4f}  F1w: {metrics['stage2_encrypted_intrusion_metrics']['f1_weighted']:.4f}\n\n"
        f"Report2:\n{metrics['stage2_encrypted_intrusion_metrics']['report_text']}",
        title="Two-Stage ARFF Pipeline Results",
        border_style="green"
    ))

if __name__ == '__main__':
    p22()


