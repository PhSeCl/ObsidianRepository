#!/usr/bin/env python3
"""自动维护 README 的「课程目录」表格（笔记数 / MOC 链接），由 pre-commit 钩子调用。

设计：
  - 单一数据源 = 仓库实际目录结构。扫描仓库根下每个「课程文件夹」（非点开头、
    含 ≥1 个 .md 的顶层目录，排除 assets/scripts/.github/.githooks 等）。
  - 笔记数 = 该文件夹内递归 .md 文件数（含该课程的 00_*_MOC.md，与历史口径一致）。
  - MOC 链接 = 文件夹内匹配 `00_*MOC*.md` 的索引文件；没有则退化为链接到文件夹。
  - 「内容概览」是人工编辑的描述，脚本**只保留、不臆造**：从现有 README 表格解析旧值
    沿用；新课程若无旧值则填占位符，提示人去补写（补好后会被永久保留）。
  - 仅替换 README 中 <!-- COURSES:START --> 与 <!-- COURSES:END --> 之间的内容，
    其余正文一字不动。排序：笔记数降序、同数按名称升序，结果稳定可复现。

零依赖（仅标准库），任何 Python3 可跑。退出码恒为 0（钩子里不应因它阻断提交）。
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, 'README.md')
START, END = '<!-- COURSES:START -->', '<!-- COURSES:END -->'
# 这些顶层目录不是课程，跳过
EXCLUDE = {'assets', 'scripts', '.github', '.githooks', '.obsidian', '.git', '.claude'}
PLACEHOLDER = '（待补充概览——请在 README 表格中补写，之后会被自动保留）'


def count_md(folder):
    n = 0
    for _dp, _dn, fns in os.walk(folder):
        n += sum(1 for f in fns if f.lower().endswith('.md'))
    return n


def find_moc(folder, name):
    """返回 (链接目标, 显示用相对路径)；优先 00_*MOC*.md，否则链接文件夹本身。"""
    cands = sorted(f for f in os.listdir(folder)
                   if re.match(r'^00_.*MOC.*\.md$', f, re.I))
    target = f'{name}/{cands[0]}' if cands else f'{name}/'
    return target


def parse_existing_overviews(text):
    """从旧 README 的标记区解析 {课程名: 内容概览}，以便沿用人工写的描述。"""
    ov = {}
    m = re.search(re.escape(START) + r'(.*?)' + re.escape(END), text, re.S)
    if not m:
        return ov
    for line in m.group(1).splitlines():
        # 形如: | [课程名](链接) | 12 | 概览文字 |
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 3:
            continue
        lm = re.match(r'\[([^\]]+)\]', cells[0])
        if lm and not cells[1].startswith(':') and cells[1] not in ('笔记数',):
            ov[lm.group(1)] = cells[2]
    return ov


def discover_courses():
    courses = []
    for name in os.listdir(ROOT):
        p = os.path.join(ROOT, name)
        if not os.path.isdir(p) or name.startswith('.') or name in EXCLUDE:
            continue
        n = count_md(p)
        if n == 0:
            continue
        courses.append({'name': name, 'count': n, 'moc': find_moc(p, name)})
    courses.sort(key=lambda c: (-c['count'], c['name']))
    return courses


def build_table(courses, overviews):
    rows = ['| 课程 | 笔记数 | 内容概览 |', '|------|:---:|------|']
    for c in courses:
        ov = overviews.get(c['name'], PLACEHOLDER)
        rows.append(f"| [{c['name']}]({c['moc']}) | {c['count']} | {ov} |")
    return '\n'.join(rows)


def main():
    if not os.path.isfile(README):
        print('[update_readme] 未找到 README.md，跳过', file=sys.stderr)
        return 0
    text = open(README, encoding='utf-8').read()
    if START not in text or END not in text:
        print('[update_readme] README 缺少 COURSES 标记，跳过（请先加锚点）', file=sys.stderr)
        return 0
    overviews = parse_existing_overviews(text)
    courses = discover_courses()
    table = build_table(courses, overviews)
    new = re.sub(re.escape(START) + r'.*?' + re.escape(END),
                 f'{START}\n{table}\n{END}', text, flags=re.S)
    if new != text:
        open(README, 'w', encoding='utf-8', newline='\n').write(new)
        print(f'[update_readme] 已刷新课程目录（{len(courses)} 门）')
    else:
        print('[update_readme] 课程目录无变化')
    return 0


if __name__ == '__main__':
    sys.exit(main())
