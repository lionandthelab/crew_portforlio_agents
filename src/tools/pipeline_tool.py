from crewai.tools import BaseTool
from typing import Optional
from src.tools.pipeline_tool_impl import run_pipeline


class RunBacktestTool(BaseTool):
    name: str = "run_quarterly_backtest"
    description: str = (
        "Run the full quarterly portfolio pipeline with default configuration and write reports. "
        "This is the baseline backtest. Inputs: start (YYYY-MM-DD), end (YYYY-MM-DD)."
    )

    def _run(self, start: str, end: str) -> str:
        return run_pipeline(start, end)

    async def _arun(self, start: str, end: str) -> str:
        return run_pipeline(start, end)
