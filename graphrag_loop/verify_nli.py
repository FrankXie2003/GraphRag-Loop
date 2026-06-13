"""验证 NLI 模型加载与 IsSup 走 NLI 路径。

用法:
    python verify_nli.py

前提:.env 配了 NLI_MODEL(默认 mDeBERTa-v3-base-xnli-multilingual-nli-2mil7),
首次跑会下载约 1.1GB 模型。
若 NLI_MODEL 留空,会看到"NLI 不可用,走 LLM 兜底"——也是合法状态,只是慢。

用 7 个测试用例覆盖三类(entailment / neutral / contradiction):
  甄士隐资助贾雨村 50 两白银 / 冬衣 → entailment
  甄士隐拒绝资助 vs 资助证据 → contradiction
  贾雨村写过书 vs 资助证据 → neutral(原文未提)
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_this = Path(__file__).resolve().parent
sys.path.insert(0, str(_this))

from models.nli import get_nli, is_available
from reflection.issup import verify
from reflection.tokens import IsSup


def main():
    print("=" * 64)
    print("验证 NLI 加载与 IsSup")
    print("=" * 64)
    nli = get_nli()
    if nli is None:
        print("NLI 不可用(.env 未配 NLI_MODEL 或加载失败)。IsSup 将走 LLM 兜底。")
    else:
        print(f"NLI 模型已加载: {nli.name}")
        print(f"标签顺序: {nli._label_order}")

    evidence = [
        "中秋夜，贾雨村对月吟诗抒志，甄士隐听后盛赞其才，当场承诺资助。"
        "他命人封五十两白银及两套冬衣赠予贾雨村，并定十九日为黄道吉日，"
        "劝其速赴神京参加次年春闱。"
    ]
    print("\n证据(前 60 字):")
    print(f"  {evidence[0][:60]}...")

    cases = [
        ("甄士隐资助贾雨村赴京赶考",         IsSup.FULL,    "entail"),
        ("甄士隐赠了贾雨村五十两白银",       IsSup.FULL,    "entail"),
        ("甄士隐赠了贾雨村三百两白银",       IsSup.NONE,    "数字相反 contradict"),
        ("甄士隐拒绝资助贾雨村",             IsSup.NONE,    "整体相反 contradict"),
        ("贾雨村写过一本小说",               IsSup.NONE,    "原文未提 neutral"),
        ("贾雨村对月吟诗",                   IsSup.FULL,    "entail"),
        ("贾雨村的家乡是浙江",               IsSup.NONE,    "原文未提 neutral"),
    ]

    print("\n--- 测试 ---")
    correct = 0
    for claim, expected, hint in cases:
        j = verify(claim, evidence)
        ok = "✓" if j.token == expected else "✗"
        if j.token == expected:
            correct += 1
        print(f"  {ok} [{j.token.value:24s}] (score={j.score:.2f}) "
              f"'{claim}'  ({hint})")

    print(f"\n通过率: {correct}/{len(cases)}")
    print("注:NLI 严格区分 entail/neutral/contradict;NONE 包含 neutral 与 contradict。")
    print("    若 NLI 把'数字相反'判 entail,说明语义识别精度不够,后续可换更强模型。")


if __name__ == "__main__":
    main()
