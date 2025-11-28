"""
CLI Interface for P22 Encrypted Traffic IDS

Terminal-based interactive interface for the microservices architecture.
Provides commands for:
- Starting/stopping services
- Processing data files
- Running predictions
- Managing outputs
"""

import click
import sys
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.configurationService import ConfigurationService
from services.dataIngestionService import DataIngestionService
from services.lstmModelService import LSTMModelService
from services.cnnModelService import CNNModelService
from services.outputManagementService import OutputManagementService


class IDSCLIManager:
    """CLI Manager for IDS Microservices."""
    
    def __init__(self, configPath: Optional[str] = None):
        """
        Initialize CLI Manager.
        
        Args:
            configPath: Path to configuration file
        """
        self.console = Console()
        self.configPath = configPath
        
        # Initialize services
        self.configService = None
        self.dataService = None
        self.lstmService = None
        self.cnnService = None
        self.outputService = None
        
        # Setup logging
        self._setupLogging()
    
    def _setupLogging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def initializeServices(self) -> bool:
        """
        Initialize all microservices.
        
        Returns:
            True if all services initialized successfully
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                
                # Configuration Service
                task = progress.add_task("Initializing Configuration Service...", total=None)
                self.configService = ConfigurationService(self.configPath)
                self.configService.start()
                progress.update(task, completed=True)
                
                # Get configurations for other services
                config = self.configService.getConfiguration()
                
                # Data Ingestion Service
                task = progress.add_task("Initializing Data Ingestion Service...", total=None)
                self.dataService = DataIngestionService(config.get('dataIngestion'))
                self.dataService.start()
                progress.update(task, completed=True)
                
                # LSTM Model Service
                task = progress.add_task("Initializing LSTM Model Service...", total=None)
                self.lstmService = LSTMModelService(config.get('lstmModel'))
                self.lstmService.start()
                progress.update(task, completed=True)
                
                # CNN Model Service
                task = progress.add_task("Initializing CNN Model Service...", total=None)
                self.cnnService = CNNModelService(config.get('cnnModel'))
                self.cnnService.start()
                progress.update(task, completed=True)
                
                # Output Management Service
                task = progress.add_task("Initializing Output Management Service...", total=None)
                self.outputService = OutputManagementService(config.get('outputManagement'))
                self.outputService.start()
                progress.update(task, completed=True)
            
            self.console.print("\n[bold green]✓ All services initialized successfully![/bold green]\n")
            return True
            
        except Exception as e:
            self.console.print(f"\n[bold red]✗ Error initializing services: {str(e)}[/bold red]\n")
            return False
    
    def shutdownServices(self):
        """Shutdown all services."""
        services = [
            ('Output Management', self.outputService),
            ('CNN Model', self.cnnService),
            ('LSTM Model', self.lstmService),
            ('Data Ingestion', self.dataService),
            ('Configuration', self.configService)
        ]
        
        for name, service in services:
            if service and service.isRunning:
                service.stop()
                self.console.print(f"[yellow]Stopped {name} Service[/yellow]")
    
    def displayServicesStatus(self):
        """Display status of all services."""
        table = Table(title="Services Status", show_header=True, header_style="bold magenta")
        table.add_column("Service", style="cyan", width=25)
        table.add_column("Status", width=15)
        table.add_column("Uptime (s)", justify="right", width=12)
        
        services = [
            ("Configuration Service", self.configService),
            ("Data Ingestion Service", self.dataService),
            ("LSTM Model Service", self.lstmService),
            ("CNN Model Service", self.cnnService),
            ("Output Management Service", self.outputService)
        ]
        
        for name, service in services:
            if service:
                health = service.healthCheck()
                status = "[green]Running[/green]" if health['status'] == 'running' else "[red]Stopped[/red]"
                uptime = f"{health['uptime']:.2f}" if health['uptime'] else "N/A"
                table.add_row(name, status, uptime)
            else:
                table.add_row(name, "[red]Not Initialized[/red]", "N/A")
        
        self.console.print(table)
    
    def processDataFile(self, filePath: str, fileType: Optional[str] = None) -> bool:
        """
        Process a data file through the ingestion service.
        
        Args:
            filePath: Path to data file
            fileType: Type of file ('csv' or 'pcap')
            
        Returns:
            True if processing successful
        """
        try:
            self.console.print(f"\n[bold]Processing file:[/bold] {filePath}")
            
            # Process through data ingestion service
            result = self.dataService.process({
                'filePath': filePath,
                'fileType': fileType
            })
            
            # Display summary
            self.console.print(f"[green]✓ File processed successfully[/green]")
            self.console.print(f"  File Type: {result['fileType']}")
            
            if result['fileType'] == 'csv':
                self.console.print(f"  Samples: {result['sampleCount']}")
                self.console.print(f"  Features: {len(result['featureNames'])}")
            else:
                self.console.print(f"  Packets: {result['packetCount']}")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]✗ Error processing file: {str(e)}[/red]")
            return False
    
    def runPrediction(
        self, 
        filePath: str, 
        modelType: str = 'both',
        aggregate: bool = True
    ) -> Optional[dict]:
        """
        Run prediction on data file.
        
        Args:
            filePath: Path to data file
            modelType: Model to use ('lstm', 'cnn', or 'both')
            aggregate: Whether to aggregate results
            
        Returns:
            Prediction results
        """
        try:
            self.console.print(f"\n[bold]Running prediction on:[/bold] {filePath}")
            
            # Process data
            with self.console.status("[bold green]Processing data..."):
                dataResult = self.dataService.process({'filePath': filePath})
            
            features = dataResult['features']
            
            lstmResult = None
            cnnResult = None
            
            # Run LSTM prediction
            if modelType in ['lstm', 'both']:
                with self.console.status("[bold blue]Running LSTM model..."):
                    lstmResult = self.lstmService.process({
                        'features': features,
                        'metadata': {'sourceFile': filePath}
                    })
                self.console.print("[green]✓ LSTM prediction complete[/green]")
            
            # Run CNN prediction
            if modelType in ['cnn', 'both']:
                with self.console.status("[bold blue]Running CNN model..."):
                    cnnResult = self.cnnService.process({
                        'features': features,
                        'metadata': {'sourceFile': filePath}
                    })
                self.console.print("[green]✓ CNN prediction complete[/green]")
            
            # Aggregate results if both models used
            if aggregate and lstmResult and cnnResult:
                with self.console.status("[bold yellow]Aggregating results..."):
                    finalResult = self.outputService.aggregateResults(
                        lstmResult,
                        cnnResult,
                        aggregationMethod='voting'
                    )
                self.console.print("[green]✓ Results aggregated[/green]")
                
                # Display final predictions
                self._displayPredictionResults(finalResult)
                return finalResult
            
            # Display individual results
            if lstmResult:
                self._displayPredictionResults(lstmResult)
            if cnnResult:
                self._displayPredictionResults(cnnResult)
            
            return lstmResult or cnnResult
            
        except Exception as e:
            self.console.print(f"[red]✗ Error running prediction: {str(e)}[/red]")
            return None
    
    def _displayPredictionResults(self, results: dict):
        """Display prediction results in a formatted way."""
        modelType = results.get('modelType', 'Unknown')
        predictions = results.get('predictions', [])
        confidence = results.get('confidence', results.get('confidences', [0])[0] if results.get('confidences') else 0)
        
        panel = Panel(
            f"[bold]Model:[/bold] {modelType}\n"
            f"[bold]Predictions:[/bold] {predictions}\n"
            f"[bold]Confidence:[/bold] {confidence:.4f}",
            title="Prediction Results",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def generateOutputReport(self):
        """Generate and display output report."""
        try:
            report = self.outputService.generateReport()
            
            table = Table(title="Output Summary", show_header=True)
            table.add_column("Output Type", style="cyan")
            table.add_column("Count", justify="right", style="magenta")
            
            for outputType, count in report['outputCounts'].items():
                table.add_row(outputType.upper(), str(count))
            
            self.console.print("\n")
            self.console.print(table)
            self.console.print(f"\n[bold]Report saved:[/bold] {report.get('reportPath', 'N/A')}")
            
        except Exception as e:
            self.console.print(f"[red]Error generating report: {str(e)}[/red]")


# CLI Commands using Click
@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """P22 Encrypted Traffic IDS - Terminal Interface"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['manager'] = IDSCLIManager(config)


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize all microservices"""
    manager = ctx.obj['manager']
    
    rprint(Panel.fit(
        "[bold cyan]P22 Encrypted Traffic IDS[/bold cyan]\n"
        "Microservices Architecture",
        border_style="cyan"
    ))
    
    if manager.initializeServices():
        manager.displayServicesStatus()
    else:
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Display status of all services"""
    manager = ctx.obj['manager']
    manager.displayServicesStatus()


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--type', '-t', type=click.Choice(['csv', 'pcap']), help='File type')
@click.pass_context
def process(ctx, file_path, type):
    """Process a data file (CSV or PCAP)"""
    manager = ctx.obj['manager']
    
    if not manager.dataService or not manager.dataService.isRunning:
        click.echo("Error: Services not initialized. Run 'init' command first.")
        sys.exit(1)
    
    manager.processDataFile(file_path, type)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--model', '-m', type=click.Choice(['lstm', 'cnn', 'both']), default='both', help='Model to use')
@click.option('--no-aggregate', is_flag=True, help='Disable result aggregation')
@click.pass_context
def predict(ctx, file_path, model, no_aggregate):
    """Run prediction on a data file"""
    manager = ctx.obj['manager']
    
    if not all([manager.dataService, manager.lstmService, manager.cnnService]):
        click.echo("Error: Services not initialized. Run 'init' command first.")
        sys.exit(1)
    
    manager.runPrediction(file_path, model, not no_aggregate)


@cli.command()
@click.pass_context
def report(ctx):
    """Generate output summary report"""
    manager = ctx.obj['manager']
    
    if not manager.outputService or not manager.outputService.isRunning:
        click.echo("Error: Output service not initialized.")
        sys.exit(1)
    
    manager.generateOutputReport()


@cli.command()
@click.pass_context
def shutdown(ctx):
    """Shutdown all services"""
    manager = ctx.obj['manager']
    manager.console.print("\n[yellow]Shutting down services...[/yellow]")
    manager.shutdownServices()
    manager.console.print("[green]All services stopped[/green]\n")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config_default.yaml', help='Output path')
@click.pass_context
def export_config(ctx, output):
    """Export default configuration to file"""
    config_service = ConfigurationService()
    config_service.exportDefaultConfig(output)
    click.echo(f"Default configuration exported to: {output}")


if __name__ == '__main__':
    cli(obj={})
