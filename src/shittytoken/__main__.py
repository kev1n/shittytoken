"""
CLI entry point: python -m shittytoken <command>

Commands:
  seed   — Seed the knowledge graph with GPU models and initial configs
  run    — Start the orchestrator (main control loop with HITL approval)
"""

import asyncio
import sys


async def seed() -> None:
    """Seed Neo4j with GPU models, LLM models, and initial configurations."""
    from .config import Settings
    from .knowledge.client import KnowledgeGraph
    from .knowledge.seed import seed as run_seed
    from .log import configure_logging

    configure_logging(component="seed")

    settings = Settings()
    kg = KnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await kg.verify_connectivity()
    await run_seed(kg._driver)
    await kg.close()
    print("Seed complete.")


async def run() -> None:
    """Start the orchestrator."""
    from .agent.orchestrator import main

    await main()


def cli() -> None:
    commands = {
        "seed": seed,
        "run": run,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python -m shittytoken <{'|'.join(commands)}>")
        print()
        print("Commands:")
        print("  seed   Seed the knowledge graph with GPU/model data")
        print("  run    Start the orchestrator (with HITL deployment approval)")
        sys.exit(1)

    asyncio.run(commands[sys.argv[1]]())


if __name__ == "__main__":
    cli()
