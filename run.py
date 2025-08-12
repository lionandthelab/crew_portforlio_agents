import argparse, os
import logging
from src.crew_run import run_crew
from src.tools.pipeline_tool_impl import run_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument("--no-llm", action="store_true", help="Bypass CrewAI and run the pipeline tool directly.")
    p.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of real market data.",
    )
    args = p.parse_args()

    logger.info(
        f"Arguments: start={args.start}, end={args.end}, no_llm={args.no_llm}, mock_data={args.mock_data}"
    )

    if args.no_llm:
        try:
            logger.info("Running pipeline directly...")
            result = run_pipeline(
                args.start, args.end, use_real_data=not args.mock_data
            )
            print(result)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    else:
        out = run_crew(args.start, args.end, use_llm=True)
        try:
            print(out)  # may be dict or string depending on version
        except Exception:
            pass
        print("Done. See reports/ for artifacts.")
