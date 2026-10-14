"""Post-training, no-gradient source-task diagnostic; never used for tuning."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from online_rl.c1_supplement_experiment import _restore_payload, _make_foundation_comparator
from online_rl.c1_experiment import augment
from online_rl.envs import NavigationEnv

def restore(path, method):
    if method in {"tensor_learned", "output_mixer", "hierarchical_output_mixer"}:
        from online_rl.c1_tensor_controls import TensorSourceBasisPolicy
        return TensorSourceBasisPolicy.from_payload(torch.load(path, map_location="cpu", weights_only=False))["policy"]
    if method == "full_context_head":
        from online_rl.c1_context_head_controls import C1ContextHeadFullPolicy
        return C1ContextHeadFullPolicy.from_payload(torch.load(path, map_location="cpu", weights_only=False))["policy"]
    return _restore_payload(path)["policy"]

def evaluate(core,seed,path):
    env=NavigationEnv()
    obs,_=env.reset(seed=seed+999)
    rng=np.random.default_rng(seed+1000)
    before=hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in core.state_dict().values())).hexdigest()
    total=0.;completed=successes=0
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f, torch.no_grad():
        w=csv.writer(f);w.writerow(["step","action","reward","done","success"])
        for step in range(512):
            p=core.get_distribution(torch.as_tensor(augment(obs,0)[None])).distribution.probs[0].numpy().astype(float)
            a=int(rng.choice(4,p=p/p.sum()))
            obs,r,term,trunc,info=env.step(a)
            total+=r;done=term or trunc
            w.writerow([step+1,a,r,int(done),int(info["success"]) if done else ""])
            if done:
                completed+=1;successes+=int(info["success"]);obs,_=env.reset()
    after=hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in core.state_dict().values())).hexdigest()
    assert before==after
    env.close()
    return {"interactions":512,"return":total,"completed_episodes":completed,"successes":successes,"success_rate":successes/max(1,completed),"route":0,"parameter_hash_unchanged":True}

def main():
    torch.set_num_threads(1)
    out=ROOT/"outputs/c1_supplement"
    config=json.loads((ROOT/"configs/c1.json").read_text())
    paths = [("main", p) for p in out.glob("step[23]*/**/formal/**/result.json")]
    for cohort in ("representation_alternatives", "context_head_alternatives"):
        paths.extend((cohort, p) for p in (out/cohort).glob("**/formal/**/result.json"))
    if len(paths) != 57:
        raise RuntimeError(f"Expected 57 formal neural runs before probing, found {len(paths)}")
    baseline=[]
    for seed in (131,149,167,181,193,211):
        core,_=_make_foundation_comparator(config,seed)
        baseline.append({"seed":seed,**evaluate(core,seed,out/"source_retention"/f"W0_s{seed}.csv")})
    rows=[]
    for cohort, path in sorted(paths, key=lambda item: str(item[1])):
        row=json.loads(path.read_text())
        if row["status"]!="complete":continue
        core=restore(path.parent/"history/agent.pt",row["method"])
        rows.append({"cohort":cohort,"method":row["method"],"control":row["control"],"tier":row["tier"],"seed":row["seed"],
            "checkpoint":str(path.parent/"history/agent.pt"),**evaluate(core,row["seed"],path.parent/"source_eval.csv")})
    report={"scope":"Post-training source NavigationEnv, root route 0, no updates; diagnostic not used for hyperparameter selection", "baseline":baseline,"runs":rows,"additional_evaluation_interactions":512*(len(rows)+len(baseline))}
    (out/"source_retention.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"source_diagnostic_runs":len(rows),"additional_evaluation_interactions":report["additional_evaluation_interactions"]}))

if __name__=="__main__":main()
