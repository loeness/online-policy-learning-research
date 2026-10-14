"""Derive held-out first-hit diagnostics from saved training trajectories."""
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/c1_supplement"

def first_hit(path,row,context):
    start=row["training"]["seen"]["steps"]
    for name,steps in row["training"]["heldout"]["schedule"]:
        if name==context:
            budget=steps
            break
        start+=steps
    window=deque(maxlen=20)
    observed=0
    with (path.parent/"oracle_stream/episodes.csv").open(newline="",encoding="utf-8") as f:
        for episode in csv.DictReader(f):
            step=int(episode["step"])
            if (episode["context"]!=context or int(episode["mixed_context"]) or
                    not start < step <= start+budget):
                continue
            observed+=1
            window.append(int(episode["success"]))
            if len(window)==20 and sum(window)>=16:
                return {"context":context,"status":"reached","context_step":step-start,
                        "capped_first_hit_step":step-start,"observed_episodes":observed,"budget":budget}
    return {"context":context,"status":"right_censored","context_step":None,
            "capped_first_hit_step":budget,"observed_episodes":observed,"budget":budget}

def main():
    rows=[]
    cohorts=[("main",p) for p in OUT.glob("step[23]*/**/formal/**/result.json")]
    for cohort in ("representation_alternatives","context_head_alternatives"):
        cohorts.extend(("fresh_alternatives",p) for p in (OUT/cohort).glob("**/formal/**/result.json"))
    if len(cohorts)!=57:
        raise RuntimeError(f"Expected 57 formal records; found {len(cohorts)}")
    for cohort,path in sorted(cohorts,key=lambda item:str(item[1])):
        row=json.loads(path.read_text(encoding="utf-8"))
        if row["status"]!="complete":continue
        for context in ("A2","B2"):
            rows.append({"cohort":cohort,"method":row["method"],"control":row["control"],
                         "tier":row["tier"],"seed":row["seed"],**first_hit(path,row,context)})
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row["cohort"],row["method"],row["control"],row["tier"],row["context"])].append(row)
    summaries=[{"cohort":key[0],"method":key[1],"control":key[2],"tier":key[3],"context":key[4],
                "mean_capped_first_hit_steps":mean(r["capped_first_hit_step"] for r in rs),
                "right_censored_runs":sum(r["status"]=="right_censored" for r in rs),"n":len(rs)}
               for key,rs in sorted(grouped.items())]
    report={"scope":"First rolling 20 completed unmixed episodes with at least 16 successes in each held-out bout. Saved training stream only; zero added interactions. Right-censored first hits use the 8192-step cap in descriptive averages, not a time-to-event estimate. No early-stopping policy or final-performance claim at that smaller budget.",
            "additional_interactions":0,"runs":rows,"summaries":summaries}
    (OUT/"interaction_efficiency.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"diagnostic_rows":len(rows),"additional_interactions":0}))

if __name__=="__main__":main()
