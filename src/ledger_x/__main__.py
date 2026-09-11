"""Run `python -m ledger_x --help` from the repository root."""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description="Ledger-X · synthetic reconciliation research workbench")
    commands = p.add_subparsers(dest="command", required=True)
    data = commands.add_parser("seed", help="Create a NEW synthetic DuckDB and independent oracle")
    data.add_argument("--db")
    data.add_argument("--count", type=int, default=100_000)
    data.add_argument("--seed", type=int, default=42)
    for name in ("ask", "tool", "serve"):
        sub = commands.add_parser(name)
        sub.add_argument("--db")
        sub.add_argument("--merchants", default="M001", help="Trusted operator scope, not a model choice")
        if name == "ask":
            sub.add_argument("question")
            sub.add_argument("--json", action="store_true", help="Print the complete audit trace")
        elif name == "tool":
            sub.add_argument("name")
            sub.add_argument("arguments", help="JSON object")
        else:
            sub.add_argument("--port", type=int, default=8090)
    args = p.parse_args()
    from ledger_x.app.data import DEFAULT_DB, generate
    db = args.db or DEFAULT_DB
    if args.command == "seed":
        print(json.dumps(generate(db, args.count, args.seed), ensure_ascii=False, indent=2))
        return
    from ledger_x.app.tools import ToolService
    service = ToolService(db, args.merchants.split(","))
    if args.command == "tool":
        result = service.execute(args.name, json.loads(args.arguments))
    elif args.command == "ask":
        from ledger_x.app.agent import ReconciliationAgent
        result = ReconciliationAgent(service, trace_dir=DEFAULT_DB.parent / "traces").run(args.question)
    else:
        from ledger_x.app.web import create_app
        import uvicorn
        uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)
        return
    if args.command == "ask" and not args.json:
        print(result["answer"] or result.get("error") or result["status"])
        print(f"\n状态：{result['status']} · 耗时：{result['elapsed_ms']/1000:.2f}s")
        print(f"审计轨迹：{DEFAULT_DB.parent / 'traces' / (result['id'] + '.json')}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "ask" and result["status"] != "answered":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
