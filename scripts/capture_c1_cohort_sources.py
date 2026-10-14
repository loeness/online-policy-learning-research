"""Capture complete supplement source hashes and the exact Step 2 full control."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/"src/online_rl/c1_continual_controls.py"
    text=source.read_text(encoding="utf-8")
    snippets=[
        "        band_conditioned: bool = True,\n",
        "        result.band_conditioned = bool(band_conditioned)\n        like = next(result.mlp_extractor.policy_net.parameters())\n        if result.band_conditioned:\n            result.band_embedding = nn.Parameter(torch.zeros((3, 64), dtype=like.dtype, device=like.device))\n        else:\n            result.register_parameter(\"band_embedding\", None)\n",
        "        if self.band_embedding is not None:\n            band = (obs[:, 0] * 8).round().long().clamp(0, 8) // 3\n            hidden_pre_activation = hidden_pre_activation + self.band_embedding[band]\n",
        "        if self.band_embedding is not None:\n            result[\"band_embedding\"] = self.band_embedding\n",
        "            \"band_embedding_bytes\": 0 if self.band_embedding is None else self.band_embedding.numel() * self.band_embedding.element_size(),\n",
        "            \"band_conditioned\": self.band_conditioned,\n",
        "            band_conditioned=bool(spec.get(\"band_conditioned\", False)),\n",
    ]
    for snippet in snippets:
        assert text.count(snippet)==1, snippet
        text=text.replace(snippet,"")
    first=next((ROOT/"outputs/c1_supplement/step2_grouping").glob("**/validation/**/config.json"))
    hashes=json.loads(first.read_text())["input_sha256"]
    expected=next(value for key,value in hashes.items() if key.replace("\\","/")=="src/online_rl/c1_continual_controls.py")
    candidates=[text.encode("utf-8"),text.replace("\n","\r\n").encode("utf-8")]
    exact=next((value for value in candidates if hashlib.sha256(value).hexdigest()==expected),None)
    if exact is None:raise RuntimeError("Historical Step 2 source reconstruction does not match its recorded SHA")
    archived=ROOT/"archive/source_versions/c1_continual_controls_step2.py"
    archived.parent.mkdir(parents=True,exist_ok=True)
    archived.write_bytes(exact)
    historical_runner=ROOT/"archive/source_versions/c1_supplement_experiment_core.py"
    core_expected=next(value for key,value in hashes.items() if key.replace("\\","/")=="src/online_rl/c1_supplement_experiment.py")
    assert hashlib.sha256(historical_runner.read_bytes()).hexdigest()==core_expected
    inputs=list((ROOT/"src/online_rl").glob("*.py"))+list((ROOT/"configs").glob("c1*.json"))+list((ROOT/"scripts").glob("*c1*.py"))+list((ROOT/"docs").glob("C1_*.md"))+[p for p in (ROOT/"archive/source_versions").glob("*") if p.is_file()]
    manifest={str(p.relative_to(ROOT)).replace("\\","/"):{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in inputs}
    out=ROOT/"outputs/c1_supplement/cohort_source_manifest.json"
    out.write_text(json.dumps({"scope":"Complete supplemental code/config hashes; final archive manifest also covers all artifacts", "step2_historical_source_sha256":expected,"core_historical_runner_sha256":core_expected,"files":manifest},indent=2),encoding="utf-8")
    print(json.dumps({"source_files":len(manifest),"step2_historical_source_verified":True,"core_historical_runner_verified":True}))

if __name__=="__main__":main()
