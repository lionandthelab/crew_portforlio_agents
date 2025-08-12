import argparse, os
from src.crew_run import run_crew
from src.tools.pipeline_tool_impl import run_pipeline

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument("--no-llm", action="store_true", help="Bypass CrewAI and run the pipeline tool directly.")
    args = p.parse_args()

    if args.no_llm:
        print(run_pipeline(args.start, args.end))
    else:
        out = run_crew(args.start, args.end, use_llm=True)
        try:
            print(out)  # may be dict or string depending on version
        except Exception:
            pass
        print("Done. See reports/ for artifacts.")
