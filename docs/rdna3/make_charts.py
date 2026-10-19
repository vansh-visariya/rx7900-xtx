#!/usr/bin/env python3
"""Draw the README charts (prefill and decode t/s per context depth) as SVG from docs/rdna3/results/*.json.

usage: python3 docs/rdna3/make_charts.py            # writes prefill.svg and decode.svg
       python3 docs/rdna3/make_charts.py --tables   # prints the README tables (markdown)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# results file, legend label, css class, label for depths without a result
SERIES = [
    ("upstream_nodraft.json", "upstream, no draft",      "s3", "not run"),
    ("upstream_dflash.json",  "upstream + DFlash2",      "s2", "GPU hang"),
    ("fork_dflash.json",      "this fork + DFlash2",     "s1", "-"),
]
FORK = 2
UPSTREAM_MAX_CTX = 196608   # the largest -c upstream fits with this model and the draft

STYLE = """<style>
  .bg { fill: #fcfcfb; }
  .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #6f6e69; }
  .grid { stroke: #e4e3df; } .axis { stroke: #b9b8b2; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .t3 { fill: #9d9c94; }
    .grid { stroke: #34332f; } .axis { stroke: #57564f; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; }
  }
  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
</style>"""


def missing_label(lab, missing):
    return "no fit" if missing == "GPU hang" and int(lab.rstrip("K")) * 1000 > UPSTREAM_MAX_CTX else missing


def load(name):
    return {r["depth_label"]: r for r in json.load(open(os.path.join(HERE, "results", name)))}


def axis_top(v):
    for step in (25, 50, 100, 200, 250, 500):
        if v / step <= 6:
            return step, step * (int(v / step) + 1)
    return 1000, 1000 * (int(v / 1000) + 1)


def bar(cls, x, w, y0, yt):
    r = min(3, (y0 - yt) / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y0:.1f} V{yt + r:.1f} Q{x:.1f},{yt:.1f} {x + r:.1f},{yt:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{yt:.1f} {x + w:.1f},{yt + r:.1f} V{y0:.1f} Z"/>')


def chart(key, title, subtitle, out):
    data = [load(f) for f, _, _, _ in SERIES]
    labels = list(data[FORK])
    W, H = 900, 440
    ml, mr, mt, mb = 60, 20, 96, 62
    pw, ph = W - ml - mr, H - mt - mb
    step, top = axis_top(max(r[key] for d in data for r in d.values() if r.get(key)) * 1.1)
    y = lambda v: mt + ph - ph * v / top
    y0 = y(0)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{title}">',
         STYLE, f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="t1" x="{ml}" y="32" font-size="19" font-weight="600">{title}</text>',
         f'<text class="t2" x="{ml}" y="54" font-size="13">{subtitle}</text>']
    lx = ml
    for _, name, cls, _ in SERIES:
        s.append(f'<rect class="{cls}" x="{lx}" y="70" width="12" height="12" rx="2"/>')
        s.append(f'<text class="t2" x="{lx + 18}" y="81" font-size="13">{name}</text>')
        lx += 18 + 7.2 * len(name) + 30
    for v in range(0, top + 1, step):
        s.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        s.append(f'<text class="t3" x="{ml - 8}" y="{y(v) + 4:.1f}" font-size="12" text-anchor="end">{v}</text>')
    s.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    s.append(f'<text class="t3" x="18" y="{mt + ph / 2:.0f}" font-size="12" text-anchor="middle" '
             f'transform="rotate(-90 18 {mt + ph / 2:.0f})">tokens per second</text>')

    gw = pw / len(labels)
    bw = min(30, (gw - 16) / len(SERIES) - 2)
    for i, lab in enumerate(labels):
        cx = ml + gw * (i + 0.5)
        x0 = cx - (len(SERIES) * (bw + 2) - 2) / 2
        vals = [d.get(lab, {}).get(key) for d in data]
        for j, ((_, _, cls, missing), v) in enumerate(zip(SERIES, vals)):
            x = x0 + j * (bw + 2)
            if not v:
                missing = missing_label(lab, missing)
                tx, ty = x + bw / 2 + 4, y0 - 6
                s.append(f'<text class="t3" x="{tx:.1f}" y="{ty:.1f}" font-size="11" text-anchor="start" '
                         f'transform="rotate(-90 {tx:.1f} {ty:.1f})">{missing}</text>')
                continue
            s.append(bar(cls, x, bw, y0, y(v)))
            if j == FORK:
                s.append(f'<text class="t1" x="{x + bw / 2:.1f}" y="{y(v) - 6:.1f}" font-size="12" font-weight="600" '
                         f'text-anchor="middle">{v:.0f}</text>')
        best_up = max([v for j, v in enumerate(vals) if j != FORK and v] or [0])
        s.append(f'<text class="t2" x="{cx:.1f}" y="{y0 + 20:.1f}" font-size="13" text-anchor="middle">{lab}</text>')
        if best_up and vals[FORK]:
            s.append(f'<text class="t1" x="{cx:.1f}" y="{y0 + 40:.1f}" font-size="12" font-weight="600" text-anchor="middle">'
                     f'{vals[FORK] / best_up:.2f}x</text>')
    s.append(f'<text class="t3" x="{W - mr}" y="{H - 8}" font-size="11" text-anchor="end">context depth (prompt tokens); '
             f'bold: this fork vs the faster upstream result</text>')
    s.append('</svg>')
    open(os.path.join(HERE, out), "w").write("\n".join(s) + "\n")


def tables():
    data = [load(f) for f, _, _, _ in SERIES]
    for key, name in (("prefill_t_s", "Prefill"), ("decode_t_s", "Decode")):
        print(f"| {name}, tokens/s | " + " | ".join(data[FORK]) + " |")
        print("|---|" + "---:|" * len(data[FORK]))
        for (_, label, _, missing), d in zip(SERIES, data):
            cells = [f"{d[l][key]:.1f}" if l in d else missing_label(l, missing) for l in data[FORK]]
            if label == SERIES[FORK][1]:
                cells = [f"**{c}**" for c in cells]
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    import sys
    if "--tables" in sys.argv:
        tables()
        sys.exit()
    meta = json.load(open(os.path.join(HERE, "results", "meta.json")))
    chart("prefill_t_s", "Prompt processing (prefill)", meta["subtitle"], "prefill.svg")
    chart("decode_t_s", "Token generation (decode)", meta["subtitle"], "decode.svg")
