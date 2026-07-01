#!/usr/bin/env python3
"""校验 Obsidian 笔记库的链接完整性，从源头堵死「死链/空节点」。

扫描目标目录下所有 .md，报告三类问题：
  1. 死链      [[目标]] / [[目标|别名]] / [[目标#锚点]] 的目标 .md 不存在
  2. 失效锚点  [[笔记#标题]] 中 #标题 在目标笔记里找不到对应标题
  3. 破图      ![](./assets/xxx.png) 引用的图片文件缺失

设计要点：
  - 按 Obsidian 短链规则，[[X]] 只要库内存在任一名为 X.md 的文件即视为可解析。
  - 反引号行内代码 `[[X]]` 是语法示例、Obsidian 不解析，自动跳过。
  - 图片正则匹配到扩展名再收尾，正确处理含 (1) 等括号的文件名（不在 ) 处截断）。
  - alt 文本用非贪婪 .*? 匹配（要求 ] 紧跟 ( 才算结束），容忍图说里出现 ]（如
    LaTeX 区间 [γx, γx+d-f]）；否则「排除右括号的字符类」会在 alt 内第一个 ]
    处截断、整行漏检（本库 p99 破图即栽在此）。
  - 容忍括号内空格 `( ./assets/x.png )`，并对 %20 等百分号编码做 unquote 还原。
  - 发现问题以退出码 1 返回，方便收尾关卡判断「必须修到 0」。

用法:  python check_links.py "<输出目录 或 仓库根>"
"""
import re, os, sys
from urllib.parse import unquote

SKIP_DIRS = {'.obsidian', '_mineru', '.git', '.venv', '__pycache__', '.trash'}


def print_safe(text):
    try:
        print(text)
    except UnicodeEncodeError:
        stream = sys.stdout
        encoding = getattr(stream, 'encoding', None) or 'utf-8'
        if hasattr(stream, 'buffer'):
            stream.buffer.write((text + '\n').encode(encoding, errors='replace'))
            stream.buffer.flush()
        else:
            stream.write((text + '\n').encode(encoding, errors='replace').decode(encoding))

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    files = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn.endswith('.md'):
                files.append(os.path.join(dp, fn))

    exist = set(os.path.splitext(os.path.basename(f))[0] for f in files)
    headings = {}
    for f in files:
        key = os.path.splitext(os.path.basename(f))[0]
        hs = set()
        for line in open(f, encoding='utf-8'):
            m = re.match(r'^#{1,6}\s+(.*?)\s*$', line)
            if m:
                hs.add(m.group(1))
        headings.setdefault(key, set()).update(hs)

    code_span = re.compile(r'`[^`]*`')                       # 去行内代码再找 wikilink
    link_re   = re.compile(r'!?\[\[([^\]]+?)\]\]')
    img_re    = re.compile(r'!\[.*?\]\(\s*(.*?\.(?:png|jpg|jpeg|gif|svg|webp))\s*\)', re.I)

    dead, bad_anchor, broken = [], [], []
    for f in files:
        base = os.path.dirname(f)
        selfname = os.path.splitext(os.path.basename(f))[0]
        for i, raw in enumerate(open(f, encoding='utf-8'), 1):
            line = code_span.sub('', raw)
            for m in link_re.finditer(line):
                inner = m.group(1)
                left = inner.split('|', 1)[0]
                tgt = left.split('#', 1)[0].strip() or selfname   # [[#锚点]] 指向本文件
                anchor = left.split('#', 1)[1].strip() if '#' in left else None
                if tgt not in exist:
                    dead.append(f"{f}:{i}\t[[{inner}]]  ->  缺文件 {tgt}.md")
                elif anchor and anchor not in headings.get(tgt, set()):
                    bad_anchor.append(f"{f}:{i}\t[[{inner}]]  ->  {tgt} 中无标题「{anchor}」")
            for m in img_re.finditer(raw):
                p = m.group(1).strip()
                if re.match(r'^[a-z][a-z0-9+.\-]*://', p, re.I):   # 跳过 http(s) 等网络图片
                    continue
                rp = unquote(p[2:] if p.startswith('./') else p.lstrip('/'))
                if not (os.path.isfile(os.path.join(base, rp)) or os.path.isfile(os.path.join(root, rp))):
                    broken.append(f"{f}:{i}\t{p}")

    def report(title, items):
        print_safe(f"\n=== {title}: {len(items)} ===")
        for it in items:
            print_safe("  " + it)

    report("死链 [[..]]（目标文件不存在）", dead)
    report("失效锚点 [[..#标题]]（标题不存在）", bad_anchor)
    report("破图（assets 图片缺失）", broken)
    total = len(dead) + len(bad_anchor) + len(broken)
    print_safe(f"\n扫描 {len(files)} 篇笔记，问题合计: {total}")
    sys.exit(1 if total else 0)

if __name__ == '__main__':
    main()
