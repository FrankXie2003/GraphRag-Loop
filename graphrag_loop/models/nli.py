"""NLI 蕴含模型封装 —— 给定 (premise, hypothesis),输出 entail/neutral/contradict 概率。

用途:S3 IsRel 可选 + L3 IsSup(段级证据验证)。
设计:惰性单例;若 .env 未配 NLI_MODEL 或加载失败,is_available()=False,
让上层(issup.py)直接走 LLM 兜底,不阻塞主流程。

与 reranker 的分工:
  reranker(cross-encoder)     → 给"query 与候选"打相关性分(粗粒度)
  NLI(蕴含)                  → 给"证据是否支撑断言"打三类概率(精细、可量化)
  二者底层都是 cross-attention,但 NLI 是有监督训练在 entailment 任务上,更适合 IsSup。
"""

from config.connections import EMBEDDING


class NLIScorer:
    name = "nli (huggingface)"

    # 大多数 NLI 模型的标签顺序;不同 checkpoint 可能不同,加载后会校准
    _DEFAULT_LABEL_ORDER = ["entailment", "neutral", "contradiction"]

    def __init__(self, model_name=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self._torch = torch
        name = model_name or EMBEDDING.nli_model
        if not name:
            raise RuntimeError("NLI_MODEL 未配置")
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSequenceClassification.from_pretrained(name)
        self._model.eval()

        # 校准标签顺序(从模型 config 读;读不到则用默认)
        id2label = getattr(self._model.config, "id2label", None) or {}
        if id2label:
            order = [str(id2label[i]).lower() for i in sorted(id2label.keys())]
            self._label_order = order
        else:
            self._label_order = self._DEFAULT_LABEL_ORDER

    def predict(self, premise, hypothesis):
        """返回 {'entailment':p_e, 'neutral':p_n, 'contradiction':p_c}。"""
        torch = self._torch
        with torch.no_grad():
            enc = self._tok(premise, hypothesis, return_tensors="pt",
                            truncation=True, max_length=512)
            logits = self._model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
        # 按 _label_order 标签名分发
        out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for label, p in zip(self._label_order, probs):
            for k in out:
                if k in label:
                    out[k] = float(p)
                    break
        return out


_singleton = None
_unavailable = False


def get_nli():
    """惰性单例;不可用返回 None(让上层走 LLM 兜底)。"""
    global _singleton, _unavailable
    if _unavailable:
        return None
    if _singleton is None:
        try:
            _singleton = NLIScorer()
        except Exception as e:
            print(f"[NLI] 不可用({e}),IsSup 将走 LLM 兜底")
            _unavailable = True
            return None
    return _singleton


def is_available():
    return get_nli() is not None
