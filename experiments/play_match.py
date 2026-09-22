"""Play the production agent against a selectable opponent and save a replay."""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
import re
import zlib
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from kaggle_environments import make

from agents.expansion_agent import make_agent
from experiments.crop_schedules import pass_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "replays"
END_DAY = 30


def configuration(seed):
    return {
        "episodeSteps": END_DAY * 24,
        "boardSize": 10,
        "startingMoney": 3000.0,
        "maxMarketOrdersPerTurn": 10,
        "turnsPerDay": 24,
        "shedCapacity": 100,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 3,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "seed": seed,
    }


def _embedded_values(source: str):
    """Evaluate only literal assembly/decompression used by submission notebooks."""
    values = {}

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return type(node.elts)(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner, method = node.func.value, node.func.attr
            if method == "join":
                separator = evaluate(owner)
                return separator.join(evaluate(node.args[0]))
            if isinstance(owner, ast.Name) and len(node.args) == 1:
                argument = evaluate(node.args[0])
                operations = {
                    ("base64", "b64decode"): base64.b64decode,
                    ("base64", "b85decode"): base64.b85decode,
                    ("gzip", "decompress"): gzip.decompress,
                    ("zlib", "decompress"): zlib.decompress,
                }
                operation = operations.get((owner.id, method))
                if operation:
                    return operation(argument)
        raise ValueError("not a supported literal expression")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return values
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = evaluate(node.value)
        except (KeyError, TypeError, ValueError, zlib.error, gzip.BadGzipFile):
            continue
    return values


def notebook_agent(notebook: Path):
    """Load an agent source embedded in a Kaggle notebook without running cells."""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    sources = []
    embedded = []
    for cell in document.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") != "code":
            continue
        match = re.match(r"^\s*%%writefile\s+(?:\S*/)?main\.py\s*\r?\n", source)
        if match:
            sources.append(source[match.end():])
        embedded.extend(_embedded_values(source).values())

    if len(sources) > 1:
        raise ValueError(f"{notebook}: found multiple %%writefile main.py cells")
    candidates = list(sources)
    if not candidates:
        for value in embedded:
            if len(value) <= 20:
                continue
            try:
                candidates.append(value.decode("utf-8") if isinstance(value, bytes) else value)
            except UnicodeDecodeError:
                continue
    candidates = sorted(set(candidates), key=len, reverse=True)
    errors = []
    for candidate in candidates:
        try:
            namespace = {"__name__": "notebook_opponent", "__file__": "main.py"}
            exec(compile(candidate, f"{notebook}:main.py", "exec"), namespace)
        except Exception as error:
            errors.append(error)
            continue
        agent = namespace.get("agent")
        if callable(agent):
            return agent, hashlib.sha256(candidate.encode()).hexdigest()

    detail = f" ({errors[-1]})" if errors else ""
    raise ValueError(f"{notebook}: could not extract a callable agent from main.py{detail}")


def resolve_opponent(spec: str):
    """Resolve ``pass``, ``random``, a Python agent, or a Kaggle notebook."""
    if spec == "pass":
        return pass_agent, "pass", {}
    if spec == "random":
        return "random", "random", {}

    path = Path(spec).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix == ".ipynb":
        agent, source_hash = notebook_agent(path)
        return agent, path.stem, {
            "opponent_file": str(path),
            "opponent_sha256": source_hash,
        }
    if path.suffix == ".py":
        return str(path), path.stem, {
            "opponent_file": str(path),
            "opponent_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    raise ValueError("opponent must be 'pass', 'random', or a .py/.ipynb file")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug or "opponent"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _planner_report(decisions):
    """Build a self-contained, interactive target-decision report."""
    payload = json.dumps(decisions, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Planner decisions</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f7fb; --panel:#fff; --text:#172033;
  --muted:#667085; --line:#d9deea; --accent:#176b55; --good:#087443; --bad:#b42318; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#10141d; --panel:#181e29; --text:#eef2f8;
  --muted:#a7b0c0; --line:#343d4d; --accent:#66d9b5; --good:#64d49b; --bad:#ff8d85; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif }}
main {{ max-width:1280px; margin:auto; padding:28px 20px 60px }}
h1 {{ margin:0 0 4px; font-size:26px }} .sub {{ color:var(--muted); margin-bottom:20px }}
.controls {{ display:flex; gap:14px; align-items:end; flex-wrap:wrap; margin-bottom:18px }}
label {{ display:grid; gap:5px; color:var(--muted); font-size:12px }}
select,input {{ min-width:150px; padding:8px 10px; border:1px solid var(--line);
  border-radius:7px; background:var(--panel); color:var(--text) }}
.count {{ margin-left:auto; color:var(--muted); padding:8px 0 }}
.day {{ margin:24px 0 8px; font-size:18px }}
details {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; margin:8px 0 }}
summary {{ cursor:pointer; display:grid; grid-template-columns:90px 90px 1fr 1.4fr;
  gap:12px; padding:12px 14px; align-items:center }}
summary::marker {{ color:var(--accent) }} .target {{ font-weight:700 }} .reason {{ color:var(--muted) }}
.none {{ color:var(--bad) }} .kept {{ color:var(--accent) }}
.table-wrap {{ overflow:auto; border-top:1px solid var(--line) }}
table {{ width:100%; border-collapse:collapse; min-width:720px }}
th,td {{ padding:9px 12px; text-align:right; border-bottom:1px solid var(--line) }}
th:first-child,td:first-child {{ text-align:left }} th {{ color:var(--muted); font-size:12px }}
tr.selected {{ background:color-mix(in srgb,var(--accent) 12%,transparent) }}
.positive {{ color:var(--good); font-weight:650 }} .negative {{ color:var(--bad) }}
.empty {{ padding:36px; text-align:center; color:var(--muted); background:var(--panel); border-radius:9px }}
@media(max-width:650px) {{ summary {{ grid-template-columns:70px 75px 1fr }} .reason {{ grid-column:1/-1 }} .count {{ margin-left:0 }} }}
</style>
</head>
<body><main>
<h1>Planner decisions</h1>
<div class="sub">Target selection by simulation step, day, and farm tile</div>
<div class="controls">
  <label>Day<select id="day"><option value="all">All days</option></select></label>
  <label>Position<input id="position" placeholder="e.g. 2,4"></label>
  <div class="count" id="count"></div>
</div>
<div id="report"></div>
</main>
<script>
const decisions={payload};
const daySelect=document.getElementById('day');
const positionInput=document.getElementById('position');
const report=document.getElementById('report');
const count=document.getElementById('count');
const fmt=new Intl.NumberFormat('en-US',{{maximumFractionDigits:2}});
const days=[...new Set(decisions.map(d=>d.day))].sort((a,b)=>a-b);
days.forEach(day=>{{const option=document.createElement('option');option.value=day;option.textContent=`Day ${{day}}`;daySelect.append(option)}});
function target(value){{return value ? `${{value[0]}}${{value[1]?' + fertilizer':''}}` : 'NONE'}}
function reason(value){{return ({{highest_score:'highest score',retained_current_within_switch_margin:'kept current target (within switch margin)',no_profitable_candidate:'no profitable candidate'}})[value]||value}}
function render(){{
  const day=daySelect.value, query=positionInput.value.replace(/\\s/g,'');
  const rows=decisions.filter(d=>(day==='all'||String(d.day)===day)&&(!query||d.position.join(',').includes(query)));
  count.textContent=`${{rows.length}} / ${{decisions.length}} decisions`;
  report.replaceChildren();
  if(!rows.length){{const empty=document.createElement('div');empty.className='empty';empty.textContent='No decisions match this filter.';report.append(empty);return}}
  let lastDay=null;
  rows.forEach(d=>{{
    if(d.day!==lastDay){{const heading=document.createElement('h2');heading.className='day';heading.textContent=`Day ${{d.day}}`;report.append(heading);lastDay=d.day}}
    const details=document.createElement('details');
    const summary=document.createElement('summary');
    const step=document.createElement('span');step.textContent=`Step ${{d.step ?? '—'}}`;
    const pos=document.createElement('span');pos.textContent=`(${{d.position.join(', ')}})`;
    const pick=document.createElement('span');pick.className='target'+(d.selected?'':' none');pick.textContent=target(d.selected);
    const why=document.createElement('span');why.className='reason'+(d.reason.includes('retained')?' kept':'');why.textContent=reason(d.reason);
    summary.append(step,pos,pick,why);details.append(summary);
    const wrap=document.createElement('div');wrap.className='table-wrap';
    const table=document.createElement('table');
    table.innerHTML='<thead><tr><th>Candidate</th><th>Market cash</th><th>Capital</th><th>Labor</th><th>Score</th><th>Status</th></tr></thead>';
    const body=document.createElement('tbody');
    [...d.candidates].sort((a,b)=>b.score-a.score).forEach(c=>{{
      const tr=document.createElement('tr');if(c.selected)tr.className='selected';
      const values=[target(c.target),fmt.format(c.market_cash),fmt.format(c.capital_cost),fmt.format(c.labor_cost),fmt.format(c.score),c.selected?'SELECTED':(c.profitable?'eligible':'rejected')];
      values.forEach((value,i)=>{{const td=document.createElement('td');td.textContent=value;if(i===4)td.className=c.score>0?'positive':'negative';tr.append(td)}});body.append(tr)
    }});
    table.append(body);wrap.append(table);details.append(wrap);report.append(details);
  }});
}}
daySelect.addEventListener('change',render);positionInput.addEventListener('input',render);render();
</script></body></html>"""


def run(opponent="pass", seed=1, output_dir=None):
    opponent_agent, opponent_name, opponent_meta = resolve_opponent(str(opponent))
    planner_decisions = []
    current = make_agent(
        END_DAY - 1, seed=seed, decision_log=planner_decisions,
    )

    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    env.run([current, opponent_agent])

    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final = env.steps[-1]
    run_dir = output_dir / (
        f"current_vs_{_slug(opponent_name)}_seed{seed}_{_timestamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    html_path = run_dir / "replay.html"
    replay_path = run_dir / "replay.json"
    result_path = run_dir / "result.json"
    planner_log_path = run_dir / "planner_decisions.json"
    planner_report_path = run_dir / "planner_decisions.html"

    html_path.write_text(
        env.render(mode="html", width=1200, height=800),
        encoding="utf-8",
    )
    replay_path.write_text(json.dumps(env.toJSON()), encoding="utf-8")
    planner_log_path.write_text(
        json.dumps(planner_decisions, indent=2), encoding="utf-8",
    )
    planner_report_path.write_text(
        _planner_report(planner_decisions), encoding="utf-8",
    )

    current_cash = float(final[0].reward)
    opponent_cash = float(final[1].reward)
    result = {
        "seed": seed,
        "opponent": opponent_name,
        "current_cash": current_cash,
        "opponent_cash": opponent_cash,
        "margin": current_cash - opponent_cash,
        "statuses": [state.status for state in final],
        "frames": len(env.steps),
        "engine": version("kaggle-environments"),
        "configuration": dict(env.configuration),
        "output_dir": str(run_dir),
        "html": str(html_path),
        "replay_json": str(replay_path),
        "planner_decisions_json": str(planner_log_path),
        "planner_decisions_html": str(planner_report_path),
        **opponent_meta,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    if result["statuses"] != ["DONE", "DONE"]:
        raise RuntimeError(f"game did not finish normally: {result['statuses']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opponent",
        default="pass",
        help="pass, random, or a path to a .py/.ipynb agent (default: pass)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.opponent, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
