#!/usr/bin/env python3
"""
CLI Interface for P22 Encrypted Traffic IDS
"""

import click
import os
import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@click.group()
@click.pass_context
def cli(ctx):
    """P22 Encrypted Traffic IDS CLI"""
    ctx.ensure_object(dict)

@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the IDS system"""
    click.echo("Initializing P22 Encrypted Traffic IDS...")
    # TODO: Add initialization logic
    click.echo("System initialized successfully!")

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.pass_context
def predict(ctx, data_file):
    """Run prediction on data file"""
    click.echo(f"Running prediction on {data_file}...")
    # TODO: Add prediction logic
    click.echo("Prediction completed!")

@cli.command()
@click.option('--config', type=click.Path(exists=True), default=str(project_root / 'ids_configuration.yaml'),
              help='Path to YAML config file.')
@click.option('--data', type=click.Path(exists=True),
              help='Path to packet-level CSV/ARFF data file (overrides config).')
@click.option('--classification_type', type=click.Choice(['binary', 'multiclass']), default='binary',
              help='Type of classification: "binary" or "multiclass".')
@click.option('--epochs', type=int, default=10, help='Number of training epochs.')
@click.option('--output_base_dir', type=click.Path(), default=str(project_root / 'outputs'),
              help='Base directory for all outputs.')
@click.pass_context
def run(ctx, config, data, classification_type, epochs, output_base_dir):
    """Run the complete IDS pipeline"""
    click.echo("Running the complete IDS pipeline...")

    # Define the path to the run_pipeline.py script
    pipeline_script = project_root / '04_Source_Code' / 'improvedCompleteWorkflow.py'

    if not pipeline_script.exists():
        click.echo(f"Error: Pipeline script not found at {pipeline_script}", err=True)
        ctx.exit(1)

    command = [
        sys.executable,  # Use the current Python interpreter
        str(pipeline_script),
        '--config', config,
        '--classification_type', classification_type,
        '--epochs', str(epochs),
        '--output_base_dir', output_base_dir
    ]

    if data:
        command.extend(['--data', data])

    click.echo(f"Executing command: {' '.join(command)}")

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        click.echo("Pipeline executed successfully!")
        if process.stdout:
            click.echo("STDOUT:\n" + process.stdout)
        if process.stderr:
            click.echo("STDERR:\n" + process.stderr)
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running pipeline: {e}", err=True)
        if e.stdout:
            click.echo("STDOUT:\n" + e.stdout, err=True)
        if e.stderr:
            click.echo("STDERR:\n" + e.stderr, err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
        ctx.exit(1)

if __name__ == '__main__':
    cli()
