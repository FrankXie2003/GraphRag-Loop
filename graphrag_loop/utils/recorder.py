"""轨迹落地 —— 把 run_phase2 的关键过程结构化记录到 logs/*.md。

为什么不直接 tee print 到文件:那样代码到处改,且 print 顺序乱(LLM 调用是流式的)。
更干净的做法:run_phase2 在关键节点显式调 RunRecorder.section/log,最后 .save() 一次写盘。
print 仍按原样输出到终端,文件落地是额外的副作用。

文件名格式:logs/YYYYMMDD-HHMMSS-<query前16字hash>.md
人类可读,便于复盘"为什么这样答"——这对终极测试(续写)尤其重要,因为续写的 query
本身就有歧义,需要事后回看检索做了什么决策。
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path


class RunRecorder:
    def __init__(self, query, log_dir=None):
        self.query = query
        self.start = datetime.now()
        self.lines = []
        self._log_dir = Path(log_dir) if log_dir else None
        self.section("Query", query)

    def section(self, title, body=""):
        """加一个二级标题节。"""
        self.lines.append(f"\n## {title}\n")
        if body:
            self.lines.append(body)
        return self

    def log(self, text):
        self.lines.append(text)
        return self

    def code_block(self, text, lang=""):
        self.lines.append(f"\n```{lang}\n{text}\n```\n")
        return self

    def kv(self, **kwargs):
        """以 key: value 列表形式记录配置。"""
        for k, v in kwargs.items():
            self.lines.append(f"- **{k}**: {v}")
        return self

    def evidence_subgraph(self, state):
        """记录证据子图(分层)。"""
        self.section("证据子图")
        self.log(f"- **实体({len(state.evidence_entities)})**: "
                 f"{', '.join(sorted(state.evidence_entities)) or '(无)'}")
        self.log(f"- **事件({len(state.evidence_events)})**:")
        for ev in state.evidence_events:
            content = (ev.get("content") or "").replace("\n", " ")
            self.log(f"  - **{ev['name']}**: {content[:200]}")
        self.log(f"\n**检索路径({len(state.evidence_edges)} 条边)**:")
        for c in state.evidence_edges:
            arrow = "->" if c.direction == "out" else "<-"
            self.log(f"  - `[{c.score:.2f}]` {c.parent} {arrow}[{c.relation}] {c.node}")
        return self

    def answer(self, label, answer):
        """记录一个答案版本(初次/回扩后)。"""
        self.section(f"答案 — {label}")
        self.code_block(answer.text, lang="")
        if getattr(answer, "is_honest_refusal", False):
            self.log("\n> 标识:**诚实拒答**(IsSup 跳过)")
        return self

    def verification(self, verified, label="最终"):
        self.section(f"IsSup 段级验证 — {label}")
        icons = {"fully_supported": "✓", "partially_supported": "△",
                 "no_support": "✗"}
        for j in verified.judgements:
            seg = j["segment"]["text"]
            if len(seg) > 80:
                seg = seg[:80] + "..."
            ic = icons.get(j["token"].value, "?")
            self.log(f"- {ic} `{j['token'].value}` "
                     f"(score={j['score']:.2f}) `{seg}` — {j['reason']}")
        self.log(f"\n**回扩轮数**: {verified.refeed_rounds}  |  "
                 f"**全部支撑**: {verified.fully_supported}")
        return self

    def save(self, log_dir=None):
        """写到 logs/YYYYMMDD-HHMMSS-<hash>.md,返回路径。"""
        log_dir = Path(log_dir) if log_dir else (self._log_dir or Path("logs"))
        log_dir.mkdir(exist_ok=True)
        ts = self.start.strftime("%Y%m%d-%H%M%S")
        h = hashlib.md5(self.query.encode("utf-8")).hexdigest()[:8]
        path = log_dir / f"{ts}-{h}.md"

        elapsed = (datetime.now() - self.start).total_seconds()
        header = (f"# Run Log\n\n"
                  f"- **Started**: {self.start.isoformat(timespec='seconds')}\n"
                  f"- **Elapsed**: {elapsed:.1f}s\n")
        path.write_text(header + "".join(line + "\n" for line in self.lines),
                        encoding="utf-8")
        return path
