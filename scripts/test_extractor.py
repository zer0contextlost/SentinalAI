import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pandas as pd
from features.extractor import extract_features, feature_names

human = pd.read_parquet("corpus/human/semeval/A__train__shard000.parquet").head(500)
ai    = pd.read_parquet("corpus/ai/semeval/A__train__shard000.parquet").head(500)

print(f"Feature count : {len(feature_names())}")
print(f"First 8       : {feature_names()[:8]}")

h_feat = extract_features(human.iloc[0]["code"], human.iloc[0]["language"])
a_feat = extract_features(ai.iloc[0]["code"],    ai.iloc[0]["language"])

print(f"\n{'Feature':<35} {'Human':>12} {'AI':>12}")
print("-" * 62)
for k in h_feat:
    hv = h_feat[k]
    av = a_feat.get(k, "N/A")
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"{k:<35} {fmt(hv):>12} {fmt(av):>12}")
