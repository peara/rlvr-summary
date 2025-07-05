"""Command-line interface for RLVR Summary."""

import logging

import click


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Set the logging level",
)
def cli(log_level: str):
    """RLVR Summary CLI - Tool-augmented summarizer using RL-VR."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@cli.command()
@click.option("--config", "-c", help="Path to training configuration file")
@click.option("--experiment", "-e", help="Experiment name for W&B tracking")
def train(config: str, experiment: str):
    """Train the RLVR model."""
    try:
        from .training import train_ppo_model

        click.echo("🚀 Starting RLVR model training...")
        if config:
            click.echo(f"Using config: {config}")
        if experiment:
            click.echo(f"Experiment name: {experiment}")

        # Run training
        training_loop = train_ppo_model(
            config_path=config,
            experiment_name=experiment,
        )

        click.echo("✅ Training completed successfully!")

    except ImportError as e:
        click.echo(f"❌ Missing dependencies for training: {e}")
        click.echo("Please install required packages: torch, transformers, verl")
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise


@cli.command()
@click.option("--config", "-c", help="Path to evaluation configuration file")
@click.option("--model", "-m", help="Path to model checkpoint")
def evaluate(config: str, model: str):
    """Evaluate the RLVR model."""
    click.echo("Evaluation functionality will be implemented in Phase A.")
    if config:
        click.echo(f"Using config: {config}")
    if model:
        click.echo(f"Model checkpoint: {model}")


@cli.command()
@click.option("--model", "-m", help="Path to model checkpoint", required=True)
@click.option("--input", "-i", help="Input text file or single text")
@click.option("--output", "-o", help="Output file for generated summaries")
def generate(model: str, input: str, output: str):
    """Generate summaries using the trained model."""
    click.echo("Generation functionality will be implemented in Phase A.")
    click.echo(f"Model: {model}")
    if input:
        click.echo(f"Input: {input}")
    if output:
        click.echo(f"Output: {output}")


if __name__ == "__main__":
    cli()
