"""محرك البحث بالمحتوى (CBIR) + تصنيف البلد. مبني على numpy فقط (بدون torch)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class SearchEngine:
    def __init__(self, embeddings: np.ndarray, meta: pd.DataFrame):
        assert len(embeddings) == len(meta)
        self.emb = embeddings.astype("float32")
        self.meta = meta.reset_index(drop=True)
        self._clf = None

    # ---- البحث ----
    def search(self, query_emb: np.ndarray, k: int = 5, exclude_idx=None):
        """ترجّع أقرب k صور. الـ embeddings متطبّعة، فالضرب النقطي = cosine."""
        sims = self.emb @ query_emb.astype("float32")
        if exclude_idx is not None:
            sims[exclude_idx] = -np.inf
        k = min(k, len(sims) - 1)
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    # ---- تصنيف البلد ----
    def fit_classifier(self, labels=None):
        labels = self.meta["country"].values if labels is None else labels
        self._clf = LogisticRegression(max_iter=2000, C=10.0)
        self._clf.fit(self.emb, labels)
        return self

    def predict_country(self, query_emb: np.ndarray):
        if self._clf is None:
            self.fit_classifier()
        probs = self._clf.predict_proba(query_emb.reshape(1, -1))[0]
        return dict(zip(self._clf.classes_, map(float, probs)))
