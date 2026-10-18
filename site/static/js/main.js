/* semCER project page. Data in static/data/*.json is precomputed:
 *   playground.json, gallery.json   scored by the released scorer (tools/build_site_playground.py, build_site_gallery.py)
 *   review_items.json               the paper's single-edit reader items (tools/build_site_review_items.py)
 *   figures.json                    values read from the paper's vector figures (tools/extract_paper_figures.py)
 *   accent_examples.json            one typical wrong-accent and wrong-letter word per language, from the dataset
 */
(() => {
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? (k.startsWith("aria-") ? "true" : "") : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) el.append(k.nodeType ? k : document.createTextNode(k));
  return el;
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt2 = (x) => (x == null ? "–" : x.toFixed(2));
const short = (x) => x.toFixed(2).replace(/^0/, "");
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const coarse = matchMedia("(pointer: coarse)").matches;
const getJSON = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p); return r.json(); });
const press = (group, btn) => $$("button", group).forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));

/* ---------- severity colours (one blue ramp, light = harmless) ---------- */
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"];
function sevColor(s) {
  const t = Math.max(0, Math.min(1, s ?? 0)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t));
  return mix(RAMP[i], RAMP[i + 1], t - i);
}
function mix(a, b, t) {
  const p = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, "0")).join("");
}
const sevInk = (s) => (s > 0.52 ? "#fff" : "#10243f");
const level = (s) => (s < 0.25 ? "harmless" : s <= 0.75 ? "partly recoverable" : "meaning lost");
const visible = (g) => (g === " " ? "␣" : g === "\n" ? "⏎" : g === "\t" ? "⇥" : g);
const showWs = (s) => s.replace(/\n/g, "⏎").replace(/^ +| +$/g, (m) => "␣".repeat(m.length));

/* ---------- tooltip (gallery only; the page never hides a key number in one) ---------- */
const tip = h("div", { class: "tip", "aria-hidden": "true" });
document.body.append(tip);
function showTip(el, html) {
  tip.innerHTML = html;
  const r = el.getBoundingClientRect();
  tip.classList.add("on");
  const w = tip.offsetWidth;
  tip.style.left = Math.max(8, Math.min(innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + "px";
  tip.style.top = (r.top - tip.offsetHeight - 8 < 8 ? r.bottom + 8 : r.top - tip.offsetHeight - 8) + "px";
}
const hideTip = () => tip.classList.remove("on");
addEventListener("scroll", hideTip, { passive: true });
function bindTip(el, html) {
  for (const ev of ["mouseenter", "focus"]) el.addEventListener(ev, () => showTip(el, html()));
  for (const ev of ["mouseleave", "blur"]) el.addEventListener(ev, hideTip);
}

/* ---------- count-up and reveal when scrolled into view ---------- */
function tween(from, to, dur, step) {
  if (reduceMotion) return step(to);
  const t0 = performance.now();
  const f = (t) => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); step(from + (to - from) * e); if (k < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}
const whenSeen = (el, fn, opts = { threshold: 0.3 }) => {
  if (!el) return;
  const o = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { o.disconnect(); fn(); } }, opts);
  o.observe(el);
};
$$("[data-count]").forEach((el) => whenSeen(el, () => {
  const end = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0), suf = el.dataset.suffix || "";
  tween(0, end, 1100, (v) => (el.textContent = v.toFixed(dec) + suf));
}));

/* ======================================================================
 * 1. Demo: break the line yourself
 * ==================================================================== */
// one-tap examples per sentence: [the text around the error, the character, what OCR reads instead]
const PICKS = {
  en: [["8:30", "8", "3"], ["Take", "T", "t"], ["tablets", "e", "c"], ["2 tablets", "2", "7"]],
  vi: [["2 viên", "2", "7"], ["thuốc", "ố", "ô"], ["sáng", "á", "a"], ["Bệnh", "B", "b"]],
  zh: [["8600", "8", "3"], ["一倍", "一", "二"], ["收入", "入", "人"], ["已达", "已", "己"]],
};
async function playground() {
  const root = $("#play");
  if (!root) return;
  const data = await getJSON("static/data/playground.json");
  const S = { s: 0, i: null, o: null, open: null };
  const tries = [];
  const sent = () => data.sentences[S.s];

  // the grapheme index of `ch` inside the first match of `ctx` in the sentence
  const locate = (sen, ctx, ch) => {
    const text = sen.graphemes.join(""), at = text.indexOf(ctx);
    if (at < 0) return -1;
    let off = 0;
    for (let i = 0; i < sen.graphemes.length; i++) {
      if (off >= at && sen.graphemes[i] === ch && off < at + ctx.length) return i;
      off += sen.graphemes[i].length;
    }
    return -1;
  };
  // the word around grapheme i, before and after the change
  const word = (sen, i, to) => {
    const g = sen.graphemes, sep = (x) => x === " " || /[，。,;!?]/.test(x);
    let lo = i, hi = i;
    const digit = (x) => /\d/.test(x);
    if (digit(g[i])) { while (lo > 0 && digit(g[lo - 1])) lo--; while (hi < g.length - 1 && digit(g[hi + 1])) hi++; }
    else if (sen.lang === "zh") lo = Math.max(0, i - 1);  // the character and the one before it
    else { while (lo > 0 && !sep(g[lo - 1])) lo--; while (hi < g.length - 1 && !sep(g[hi + 1])) hi++; }
    const tidy = (t) => t.trim().replace(/[.。]$/, "");
    const before = tidy(g.slice(lo, hi + 1).join("")), after = tidy(g.slice(lo, i).join("") + to + g.slice(i + 1, hi + 1).join(""));
    return [before || visible(g[i]), after || "∅"];
  };

  const tabs = $("#play-tabs");
  data.sentences.forEach((x, k) => tabs.append(h("button", {
    class: "chip", type: "button", "aria-pressed": "false",
    onclick: (e) => { S.s = k; S.i = S.o = S.open = null; tries.length = 0; press(tabs, e.currentTarget); render(); },
  }, x.name)));

  function choose(i, j, label) {
    S.i = i; S.o = j; S.open = null;
    const sen = sent(), o = sen.options[i][j], [w0, w1] = label || word(sen, i, o.to);
    const key = `${sen.lang}.${i}.${j}`;
    const old = tries.findIndex((t) => t.key === key);
    if (old >= 0) tries.splice(old, 1);
    tries.unshift({ key, w0, w1, sev: o.severity, n: sen.graphemes.length });
    tries.length = Math.min(tries.length, 6);
    render();
  }

  document.addEventListener("click", (e) => { if (S.open != null && !e.target.closest(".pl-ref, .pl-pop")) { S.open = null; render(); } });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && S.open != null) { S.open = null; render(); } });
  $("#play-ref").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const gs = $$(".g", e.currentTarget), i = gs.indexOf(document.activeElement);
    gs[Math.max(0, Math.min(gs.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)))]?.focus(); e.preventDefault();
  });

  function render() {
    const sen = sent();
    press(tabs, $$("button", tabs)[S.s]);
    const o = S.i != null && S.o != null ? sen.options[S.i][S.o] : null;

    // one-tap examples
    $("#play-picks").replaceChildren(...(PICKS[sen.lang] || []).map(([ctx, from, to]) => {
      const i = locate(sen, ctx, from), j = i < 0 ? -1 : sen.options[i].findIndex((x) => x.to === to);
      if (j < 0) return null;
      const k = ctx.indexOf(from), w0 = ctx, w1 = ctx.slice(0, k) + to + ctx.slice(k + from.length);  // labelled with its own text
      return h("button", { class: "chip pick" + (S.i === i && S.o === j ? " on" : ""), type: "button", onclick: () => choose(i, j, [w0, w1]) },
        h("s", {}, w0), " → ", w1);
    }).filter(Boolean));

    // the original, one button per character
    const ref = $("#play-ref");
    ref.replaceChildren();
    ref.setAttribute("lang", sen.lang);
    let w = null;
    sen.graphemes.forEach((g, i) => {
      const b = h("button", {
        class: "g" + (g === " " ? " sp" : "") + (S.i === i ? " sel" : "") + (S.open === i ? " open" : ""),
        type: "button", "aria-label": `misread ${g === " " ? "the space" : g}`, "aria-expanded": String(S.open === i),
        onclick: (e) => { e.stopPropagation(); S.open = S.open === i ? null : i; render(); $(".pl-pop .opt")?.focus({ preventScroll: true }); },
      }, g === " " ? " " : g);
      if (g === " " || sen.lang === "zh") { ref.append(b); w = null; }
      else { if (!w) { w = h("span", { class: "w" }); ref.append(w); } w.append(b); }
    });

    // the options, right under the tapped character
    const pop = $("#play-pop");
    if (S.open == null) pop.hidden = true;
    else {
      const g = sen.graphemes[S.open];
      // letters are small targets on a phone: step to the neighbour without re-aiming
      const step = (d) => h("button", { type: "button", class: "pop-step", "aria-label": d < 0 ? "previous letter" : "next letter",
        disabled: !sen.graphemes[S.open + d], onclick: (e) => { e.stopPropagation(); S.open += d; render(); } }, d < 0 ? "‹" : "›");
      pop.replaceChildren(step(-1), step(1), h("span", { class: "pop-sep" }), h("span", { class: "pop-lbl" }, `OCR reads “${visible(g)}” as`),
        ...sen.options[S.open].map((x, j) => h("button", { type: "button", class: "opt", onclick: (e) => { e.stopPropagation(); choose(S.open, j); } },
          x.to === "" ? h("span", { class: "del-glyph" }, "nothing") : visible(x.to))));
      pop.hidden = false;
      const btn = $$(".g", ref)[S.open], box = $(".play-lines").getBoundingClientRect(), r = btn.getBoundingClientRect();
      pop.style.top = r.bottom - box.top + 6 + "px";
      pop.style.left = Math.max(0, Math.min(box.width - pop.offsetWidth, r.left - box.left + r.width / 2 - 28)) + "px";
    }
    $("#play-gloss").textContent = sen.gloss || "";
    $("#play-gloss").hidden = !sen.gloss;

    // what OCR read
    const hyp = $("#play-hyp");
    hyp.replaceChildren();
    sen.graphemes.forEach((g, i) => hyp.append(o && i === S.i
      ? h("span", { class: "e pop" + (o.to === "" ? " del" : ""), style: `background:${sevColor(o.severity)};color:${sevInk(o.severity)}` }, o.to === "" ? " " : visible(o.to))
      : g));

    // the two scores
    const n = sen.graphemes.length;
    $("#s-cer").textContent = "1 error";
    $("#s-cer-n").textContent = o ? `${pct(1 / n, 1)} of this line. The same for any single error.` : "Pick an error to see the scores.";
    $("#s-sem").textContent = o ? o.severity.toFixed(2) : "–";
    $("#s-sem").dataset.level = o ? level(o.severity) : "";
    $("#s-sem-n").innerHTML = o ? `<b>${level(o.severity)}</b> · ${pct(o.severity / n, 1)} of this line` : "&nbsp;";
    $("#gauge-needle").style.left = 100 * (o ? o.severity : 0) + "%";
    $("#gauge-needle").style.opacity = o ? 1 : 0;
    root.classList.toggle("has-error", !!o);

    // the running list
    $("#tries").hidden = tries.length < 1;
    $("#tries-list").replaceChildren(...tries.map((t, k) => h("li", { class: k === 0 ? "now" : "" },
      h("span", { class: "tw" }, h("s", {}, t.w0), " → ", t.w1),
      h("span", { class: "tc" }, "1 error"),
      h("span", { class: "ts" }, h("span", { class: "tbar" }, h("i", { style: `width:${100 * t.sev}%;background:${sevColor(t.sev)}` })), h("b", {}, t.sev.toFixed(2))))));
  }
  render();
  $("#play-picks .pick")?.click();  // open on the first example, so the scores are never empty
  addEventListener("resize", () => S.open != null && render());
}

/* ======================================================================
 * 2. Method: one line, priced twice (the judge's charges, from the paper)
 * ==================================================================== */
function priceTag() {
  const root = $("#price");
  if (!root) return;
  const same = (t) => [...t].map((c) => [c, c]);
  // [reference, OCR, cost, note]; a missing side is an insertion or a deletion
  const COLS = [
    ["F", "f", 0.00, 1], ...same("uture."), [null, " ", 0.00, 2], ...same("We have deter"),
    ["m", null, 0.28, 3], ["i", null, 0.43, 3], ["n", "t", 0.79, 3], ...same("ed"), [" ", null, 0.40, 4], ...same(" that the"),
  ];
  const NOTES = [
    { title: "F → f", text: "The scan prints a lowercase f too. Nothing lost." },
    { title: "An extra space", text: "Easy to read past." },
    { title: "determined → deterted", text: "One wrong word, split into three edits. This is where the meaning goes." },
    { title: "A missing space", text: "Two words run together. Readable, with effort." },
  ];
  const nRef = COLS.filter((c) => c[0] != null).length;
  const edits = COLS.filter((c) => c.length === 4);
  const semSum = edits.reduce((a, c) => a + c[2], 0);
  const line = $("#pr-line");
  line.style.setProperty("--cols", COLS.length);
  const seen = new Set();
  COLS.forEach(([r, o, sev, note], i) => {
    const col = `grid-column:${i + 1}`;
    const hit = note !== undefined;
    if (hit && !seen.has(note)) { seen.add(note); line.append(h("span", { class: "pin", style: `${col};grid-row:1` }, note)); }
    const cell = (g, row) => h("span", { class: "pc" + (g == null ? " gap" : g === " " ? " ws" : "") + (hit ? " hit" : ""), style: `${col};grid-row:${row}` }, g == null ? "·" : g === " " ? "␣" : g);
    line.append(cell(r, 2), cell(o, 3));
    if (hit) line.append(h("span", { class: "tag", style: `${col};grid-row:4;--sev:${sevColor(sev)};--ink:${sevInk(sev)}` },
      h("span", { class: "tag-cer" }, "1"), h("span", { class: "tag-sem" }, short(sev))));
  });
  $("#pr-notes").append(...NOTES.map((n, k) => {
    const cols = edits.filter((c) => c[3] === k + 1);
    const cost = cols.reduce((a, c) => a + c[2], 0);
    return h("li", {}, h("span", { class: "pin" }, k + 1),
      h("div", {}, h("b", {}, n.title), h("p", {}, n.text)),
      h("span", { class: "np" }, h("span", { class: "np-cer" }, String(cols.length)), h("span", { class: "np-sem" }, short(cost))));
  }));
  const seg = $(".seg", root);
  let shown = semSum / nRef;
  function set(mode, animate = true) {
    root.dataset.mode = mode;
    press(seg, $(`[data-mode=${mode}]`, seg));
    const to = mode === "cer" ? edits.length / nRef : semSum / nRef;
    $("#price-total-k").textContent = mode === "cer" ? "CER" : "semCER";
    $("#price-total-eq").textContent = mode === "cer" ? `${edits.length} × 1 ÷ ${nRef} characters` : `${semSum.toFixed(2)} ÷ ${nRef} characters`;
    tween(animate ? shown : to, to, 700, (v) => { $("#price-total").textContent = pct(v, 1); shown = v; });
  }
  $$("button", seg).forEach((b) => b.addEventListener("click", () => set(b.dataset.mode)));
  set("cer", false);
  // on a narrow screen the line scrolls; start it at the wrong word, where the cost is
  const sc = $(".pr-scroll", root), focus = $$(".pin", line)[2];
  const centre = () => { if (sc.scrollWidth > sc.clientWidth) sc.scrollLeft = focus.offsetLeft - sc.clientWidth / 2 + 40; };
  centre(); addEventListener("resize", centre);
  whenSeen(root, () => setTimeout(() => set("sem"), reduceMotion ? 0 : 1100), { threshold: 0.4 });
}

/* ======================================================================
 * 3. Charts (uPlot, loaded when the results come near)
 * ==================================================================== */
const charts = [];
const T = () => ({
  ink: css("--strong"), text: css("--text"), muted: css("--muted"), line: css("--line"),
  sem: css("--accent"), teal: css("--teal"), cer: css("--cer"), faint: "#c9c9c4",
  font: `12px ${css("--sans")}`, fontLg: `13px ${css("--sans")}`,
});
const axis = (t, extra = {}) => ({ stroke: t.muted, font: t.font, labelFont: t.fontLg, grid: { stroke: t.line, width: 1 }, ticks: { stroke: t.line, width: 1, size: 4 }, ...extra });
function fit(u, el, height) {
  const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height: typeof height === "function" ? height() : height }));
  ro.observe(el);
  return ro;
}
// uPlot follows a mouse; on a phone, a tap or a sideways drag moves the cursor and the page still scrolls
function touchCursor(u) {
  const over = u.over;
  over.style.touchAction = "pan-y";
  const move = (e) => {
    if (e.pointerType === "mouse") return;
    const r = over.getBoundingClientRect();
    u.setCursor({ left: e.clientX - r.left, top: e.clientY - r.top });
  };
  over.addEventListener("pointerdown", move);
  over.addEventListener("pointermove", move);
}
// keep the last position when the pointer leaves, instead of blanking the readout
const sticky = (onIdx) => (u) => { if (u.cursor.idx != null) onIdx(u.cursor.idx); };

/* ---- result 1: damage found, and the real errors behind each order ---- */
function reviewChart(fig, items) {
  const el = $("#chart-review");
  if (!el) return;
  const LANGS = { en: "English", vi: "Vietnamese", zh: "Chinese" };
  const SER = [
    { key: "readers", label: "readers' own picks", color: "ink", width: 2 },
    { key: "severity", label: "sorted by semCER", color: "sem", width: 3.5 },
    { key: "judge_free", label: "best rule without semCER", color: "teal", width: 2 },
    { key: "cer", label: "sorted by CER", color: "cer", width: 3 },
    { key: "random", label: "random", color: "faint", width: 2, dash: [5, 5] },
  ];
  let lang = "en", u, ro, idx = 10;
  const xs = () => fig.review[lang].readers.map((p) => Math.min(50, p[0]));
  const at20 = () => xs().findIndex((x) => Math.round(x) === 20);
  const val = (key, i) => fig.review[lang][key][i][1];
  const dataFor = () => {
    return [xs(), ...SER.map((s) => fig.review[lang][s.key].map((p) => p[1]))];
  };
  const orders = () => {
    const list = items[lang];
    return {
      sem: [...list].sort((a, b) => b.severity - a.severity),
      cer: [...list].sort((a, b) => b.cer - a.cer),  // CER charges 1/length, so short lines come first
    };
  };

  const tabs = $("#review-tabs");
  Object.entries(LANGS).forEach(([k, name]) => tabs.append(h("button", {
    class: "chip", type: "button", "aria-pressed": String(k === lang),
    onclick: (e) => { lang = k; press(tabs, e.currentTarget); u.setData(dataFor()); lanes(idx); },
  }, name)));

  // examples shown under the chart: skip symbol-only items, which read as noise (the curve still counts them)
  const wordy = (it) => /[\p{L}\p{N}]{2,}/u.test(it.ref) && /[\p{L}\p{N}]/u.test(it.hyp);
  function chip(it) {
    return h("li", { title: `readers rated it ${fmt2(it.readers)}` },
      h("span", { class: "lw" }, h("s", {}, showWs(it.ref)), " → ", h("span", {}, showWs(it.hyp))),
      h("span", { class: "lr", style: `--r:${it.readers}` }, h("i")));
  }
  function lanes(i) {
    const box = $("#review-lanes");
    const x = xs()[i], n = items[lang].length, k = Math.round((x / 100) * n), o = orders();
    const lane = (key, label, cls) => h("div", { class: `lane ${cls}` },
      h("div", { class: "lane-head" }, h("span", {}, label), h("span", {}, h("b", {}, pct(val(key, i), 0)), " of the damage found")),
      h("ol", {}, o[cls].slice(0, k).reverse().filter(wordy).slice(0, 4).map(chip)));
    box.replaceChildren(
      h("p", { class: "lanes-head" }, `After checking ${Math.round(x)}% of the errors (${k} of ${n}), some of the latest each order checked:`),
      lane("severity", "sorted by semCER", "sem"), lane("cer", "sorted by CER", "cer"),
      h("p", { class: "lanes-key" }, h("i"), "how much readers said it hurt"));
  }

  function build() {
    const t = T();
    u?.destroy(); ro?.disconnect();
    u = new uPlot({
      width: el.clientWidth, height: 300,
      padding: [14, 24, 0, 0],
      scales: { x: { time: false, range: [0, 50] }, y: { range: [0, 0.92] } },
      legend: { show: false },
      cursor: { y: false, points: { size: 9, width: 2, fill: "#fff" }, drag: { x: false, y: false } },
      axes: [
        axis(t, { label: "errors checked", values: (u, v) => v.map((x) => x + "%") }),
        axis(t, { label: "damage found", size: 52, values: (u, v) => v.map((x) => Math.round(x * 100) + "%") }),
      ],
      series: [{}, ...SER.map((s) => ({ label: s.label, stroke: t[s.color], width: s.width, dash: s.dash, points: { show: false }, spanGaps: false }))],
      hooks: {
        setCursor: [sticky((i) => { if (i !== idx) { idx = i; lanes(i); } u.redraw(false, false); })],
        draw: [(u) => {
          const c = u.ctx, x20 = u.valToPos(20, "x", true), dpr = devicePixelRatio;
          c.save();
          c.strokeStyle = t.muted; c.globalAlpha = 0.5; c.setLineDash([3 * dpr, 4 * dpr]); c.lineWidth = dpr;
          c.beginPath(); c.moveTo(x20, u.bbox.top); c.lineTo(x20, u.bbox.top + u.bbox.height); c.stroke();
          c.setLineDash([]); c.globalAlpha = 1;
          if (u.cursor.idx != null) {  // the two values that matter, written next to their points
            const i = u.cursor.idx, x = u.valToPos(xs()[i], "x", true);
            c.font = `700 ${13 * dpr}px ${css("--sans")}`; c.textAlign = "left";
            c.textAlign = "right"; c.lineWidth = 4 * dpr; c.strokeStyle = "#fff"; c.lineJoin = "round";
            const tag = (txt, col, y) => { c.strokeText(txt, x - 9 * dpr, y); c.fillStyle = col; c.fillText(txt, x - 9 * dpr, y); };
            tag(pct(val("severity", i), 0), t.sem, u.valToPos(val("severity", i), "y", true) - 8 * dpr);
            tag(pct(val("cer", i), 0), t.cer, u.valToPos(val("cer", i), "y", true) + 18 * dpr);
          }
          {  // label the two lines that matter at their ends
            const n = xs().length - 1;
            c.textAlign = "right"; c.font = `600 ${12 * dpr}px ${css("--sans")}`;
            c.fillStyle = t.sem; c.fillText("semCER", u.valToPos(49, "x", true), u.valToPos(val("severity", n), "y", true) + 18 * dpr);
            c.fillStyle = t.cer; c.fillText("CER", u.valToPos(49, "x", true), u.valToPos(val("cer", n), "y", true) + 18 * dpr);
          }
          c.restore();
        }],
      },
    }, dataFor(), el);
    ro = fit(u, el, 300);
    touchCursor(u);
    const keys = $("#review-keys");
    keys.replaceChildren(...SER.map((s, i) => h("button", {
      type: "button", class: "key on", "aria-pressed": "true",
      onclick: (e) => { const b = e.currentTarget, on = b.classList.toggle("on"); b.setAttribute("aria-pressed", String(on)); u.setSeries(i + 1, { show: on }); },
    }, h("i", { style: s.dash ? `background:repeating-linear-gradient(90deg,${t[s.color]} 0 5px,transparent 5px 9px)` : `background:${t[s.color]}` }), s.label)));
    idx = at20(); lanes(idx);
    requestAnimationFrame(() => u.setCursor({ left: u.valToPos(xs()[idx], "x"), top: u.valToPos(val("severity", idx), "y") }));
  }
  build();
  charts.push(build);
}

/* ---- result 3: a wrong accent's cost as a share of a wrong letter's ---- */
const SCRIPTS = { alphabet: "#0f9a94", abjad: "#c9821a", abugida: "#8a5cd6" };  // validated palette
function accentBars(fig, examples) {
  const root = $("#acc");
  if (!root) return;
  const rows = fig.load.filter((r) => r.mark_only != null).map((r) => ({ ...r, ratio: r.mark_only / r.content_sub }));
  const els = new Map();
  for (const r of rows) {
    const ex = (examples[r.lang] || {}).mark;
    const fill = h("span", { class: "ab-fill", "data-w": r.ratio });
    const el = h("div", { class: "ab", role: "listitem", style: `--c:${SCRIPTS[r.script]}`, title: `${r.name}: wrong accent ${fmt2(r.mark_only)}, wrong letter ${fmt2(r.content_sub)}` },
      h("span", { class: "ab-name" }, r.name),
      h("span", { class: "ab-track" }, fill), h("b", { class: "ab-v" }, pct(r.ratio, 0)),
      h("span", { class: "ab-ex" }, ex ? [h("span", { class: "mono" }, h("s", {}, ex.ref), " → ", ex.hyp), h("small", {}, ` costs ${fmt2(ex.severity)}`)] : h("small", {}, `accent ${fmt2(r.mark_only)} · letter ${fmt2(r.content_sub)}`)));
    els.set(r.name, el);
  }
  const ORDER = ["alphabet", "abjad", "abugida"];
  function place(sort) {
    const before = new Map([...els].map(([k, el]) => [k, el.getBoundingClientRect().top]));
    const list = [...rows].sort((x, y) => (sort === "script" ? ORDER.indexOf(x.script) - ORDER.indexOf(y.script) : 0) || x.ratio - y.ratio);
    root.replaceChildren();
    let last = null;
    for (const r of list) {
      if (sort === "script" && r.script !== last) root.append(h("div", { class: "ab-group", style: `--c:${SCRIPTS[r.script]}` }, r.script === "abugida" ? "abugidas (Hindi, Thai, Telugu…)" : r.script === "abjad" ? "abjads (Arabic, Urdu)" : "alphabets"));
      last = r.script;
      root.append(els.get(r.name));
    }
    if (reduceMotion) return;
    for (const [k, el] of els) {  // slide rows from their old place to the new one
      const dy = before.get(k) - el.getBoundingClientRect().top;
      if (!dy || !before.get(k)) continue;
      el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 450, easing: "cubic-bezier(.2,.8,.2,1)" });
    }
  }
  place("ratio");
  $("#acc-key").append(...ORDER.map((k) => h("span", { class: "key static" }, h("i", { class: "shape circle", style: `--c:${SCRIPTS[k]}` }), k)),
    h("span", { class: "muted" }, "Chinese writes no accents."));
  whenSeen(root, () => $$(".ab-fill", root).forEach((f, i) => setTimeout(() => (f.style.width = 100 * f.dataset.w + "%"), reduceMotion ? 0 : i * 35)));
}

/* ---- result 4: classic OCR to VLM, one row per language, as two charts ---- */
function vlmArrows(fig) {
  const svgs = [$("#vlm-a"), $("#vlm-b")];
  if (!svgs[0]) return;
  const NS = "http://www.w3.org/2000/svg";
  const rows = fig.system.filter((r) => r.classical && r.general_vlm).sort((a, b) => b.general_vlm.real_word - a.general_vlm.real_word);
  const PANELS = [
    { key: "real_word", lo: 0, hi: 100, ticks: [0, 25, 50, 75, 100], fmt: (v) => v + "%" },
    { key: "severity", lo: 0.4, hi: 0.9, ticks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fmt: (v) => v.toFixed(1) },
  ];
  let on = rows.findIndex((r) => r.lang === "vi"), seen = false;
  const S = (tag, a = {}, text) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (text != null) e.textContent = text; return e; };
  function say() {
    const r = rows[on], c = r.classical, g = r.general_vlm;
    $("#vlm-readout").innerHTML = `<b>${r.name}</b><span class="ro-items"><span><b class="qn sm">A</b>real-word errors <b class="c-cer">${Math.round(c.real_word)}%</b> → <b class="c-sem">${Math.round(g.real_word)}%</b></span><span><b class="qn sm">B</b>cost <b class="c-cer">${fmt2(c.severity)}</b> → <b class="c-sem">${fmt2(g.severity)}</b></span>${g.severity < c.severity ? `<span class="muted">Spanish is the one language where the cost goes down.</span>` : ""}</span>`;
    svgs.forEach((svg) => $$(".vr", svg).forEach((el) => el.classList.toggle("dim", +el.dataset.i !== on)));
  }
  function draw(svg, p) {
    const W = svg.clientWidth || 440, NAME = 92, TOP = 8, ROW = 36, AX = 30;
    const H = TOP + rows.length * ROW + AX, pw = W - NAME - 16;
    const X = (v) => NAME + ((v - p.lo) / (p.hi - p.lo)) * pw;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H);
    svg.replaceChildren();
    const defs = S("defs"), id = "vah-" + p.key;
    const mk = S("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 9, markerHeight: 9, markerUnits: "userSpaceOnUse", orient: "auto" });
    mk.append(S("path", { d: "M0 0 L10 5 L0 10 z", class: "vh" }));
    defs.append(mk); svg.append(defs);
    const g = S("g", { class: "va-axes" });
    for (const t of p.ticks) g.append(S("line", { x1: X(t), x2: X(t), y1: TOP, y2: H - AX + 4 }), S("text", { x: X(t), y: H - AX + 22, "text-anchor": "middle" }, p.fmt(t)));
    svg.append(g);
    rows.forEach((r, i) => {
      const y = TOP + i * ROW + ROW / 2;
      const grp = S("g", { class: "vr", "data-i": i, tabindex: 0, role: "button", "aria-label": r.name });
      const a = X(r.classical[p.key]), b = X(r.general_vlm[p.key]), dir = Math.sign(b - a) || 1, len = Math.abs(b - a);
      const line = S("line", { x1: a, y1: y, x2: b - dir * 8, y2: y, class: "va-line" + (dir < 0 ? " down" : ""), "marker-end": `url(#${id})` });
      line.style.strokeDasharray = len; line.style.strokeDashoffset = seen || reduceMotion ? 0 : len;
      grp.append(S("rect", { x: 0, y: y - ROW / 2, width: W, height: ROW, class: "vr-bg" }),
        S("text", { x: NAME - 12, y: y + 5, "text-anchor": "end", class: "va-name" }, r.name),
        line, S("rect", { x: a - 6, y: y - 6, width: 12, height: 12, rx: 2, class: "va-from" }), S("circle", { cx: b, cy: y, r: 7, class: "va-to" }));
      const pick = () => { on = i; say(); };
      grp.addEventListener("pointerenter", pick); grp.addEventListener("focus", pick); grp.addEventListener("click", pick);
      svg.append(grp);
    });
  }
  const all = () => { svgs.forEach((svg, k) => draw(svg, PANELS[k])); say(); };
  all();
  svgs.forEach((svg) => new ResizeObserver(all).observe(svg));
  whenSeen(svgs[0], () => {
    svgs.forEach((svg, k) => $$(".va-line", svg).forEach((l, i) => l.animate([{ strokeDashoffset: l.style.strokeDasharray }, { strokeDashoffset: 0 }],
      { duration: reduceMotion ? 0 : 650, delay: k * 450 + i * 70, easing: "cubic-bezier(.3,.7,.2,1)", fill: "forwards" })));
    seen = true;
  });
}

/* ---- result 2: agreement bars ---- */
function agreement() {
  const root = $("#agree");
  if (!root) return;
  const ROWS = [
    { lang: "English", sem: 0.71, cer: 0.58, n: 173 },
    { lang: "Vietnamese", sem: 0.79, cer: 0.80, n: 163 },
    { lang: "Chinese", sem: 0.78, cer: 0.60, n: 146 },
  ];
  ROWS.forEach((r) => root.append(h("div", { class: "agree-row" },
    h("div", { class: "agree-lang" }, r.lang, h("small", {}, `${r.n} pairs`)),
    h("div", { class: "bar" }, h("div", { class: "fill-wrap" }, h("div", { class: "fill sem", "data-w": r.sem })), h("span", { class: "v" }, pct(r.sem, 0))),
    h("div", { class: "bar" }, h("div", { class: "fill-wrap" }, h("div", { class: "fill cer", "data-w": r.cer })), h("span", { class: "v" }, pct(r.cer, 0))),
  )));
  whenSeen(root, () => $$(".fill", root).forEach((f) => (f.style.width = 100 * f.dataset.w + "%")));
}

function loadScript(src) {
  return new Promise((ok, fail) => { const s = h("script", { src }); s.onload = ok; s.onerror = fail; document.head.append(s); });
}
function results() {
  const sec = $("#findings");
  if (!sec) return;
  const go = async () => {
    document.head.append(h("link", { rel: "stylesheet", href: "static/vendor/uplot/uPlot.min.css" }));
    const [fig, items] = await Promise.all([
      getJSON("static/data/figures.json"), getJSON("static/data/review_items.json"),
      loadScript("static/vendor/uplot/uPlot.iife.min.js"),
    ]);
    reviewChart(fig, items);
    document.fonts?.ready.then(() => charts.forEach((b) => b()));  // label widths change once fonts arrive
  };
  whenSeen(sec, () => go().catch(console.error), { rootMargin: "800px 0px" });
  // the bar and arrow charts need no uPlot
  Promise.all([getJSON("static/data/figures.json"), getJSON("static/data/accent_examples.json")])
    .then(([fig, ex]) => { accentBars(fig, ex); vlmArrows(fig); }).catch(console.error);
}

/* ======================================================================
 * 4. Gallery page
 * ==================================================================== */
async function gallery() {
  const root = $("#cards");
  if (!root) return;
  const data = await getJSON("static/data/gallery.json");
  const NAMES = { en: "English", de: "German", fr: "French", es: "Spanish", pt: "Portuguese", ru: "Russian", ur: "Urdu" };
  const langs = [...new Set(data.items.map((x) => x.lang))];
  let on = "all", cerView = false;
  const filters = $("#filters");
  ["all", ...langs].forEach((l) => filters.append(h("button", {
    class: "chip", type: "button", "aria-pressed": String(l === on),
    onclick: (e) => { on = l; press(filters, e.currentTarget); draw(); },
  }, l === "all" ? "All languages" : NAMES[l] || l)));
  $("#cer-view").addEventListener("change", (e) => { cerView = e.target.checked; root.classList.toggle("cer-view", cerView); });

  function marked(text, edits, side, lang) {
    const spans = edits.filter((e) => side === "hyp" || e.op !== "INS").map((e) => ({ e, s: side === "hyp" ? e.hyp_span : e.ref_span })).sort((a, b) => a.s[0] - b.s[0]);
    const out = h("span", { class: "txt", dir: "auto", lang });
    const ZWJ = ["ur", "ar", "fa"].includes(lang) ? "\u200d" : "";  // keeps cursive letters joined across a highlight
    let at = 0;
    for (const { e, s } of spans) {
      out.append(text.slice(at, s[0]) + ZWJ);
      if (side === "ref") out.append(h("span", { class: "e ref" }, ZWJ + visible(text.slice(s[0], s[1])) + ZWJ));
      else {
        const del = e.op === "DEL";
        const el = h("span", { class: "e" + (del ? " del" : ""), tabindex: 0, style: `--sev:${sevColor(e.severity)};--ink:${sevInk(e.severity)}` }, del ? "" : ZWJ + visible(text.slice(s[0], s[1])) + ZWJ);
        const what = e.op === "SUB" ? `“${visible(e.ref)}” → “${visible(e.hyp)}”` : del ? `“${visible(e.ref)}” deleted` : `“${visible(e.hyp)}” inserted`;
        bindTip(el, () => (cerView ? `${what}<br>CER charge <b>1</b>` : `${what}<br>cost <b>${fmt2(e.severity)}</b> · ${level(e.severity)}`));
        out.append(el);
      }
      at = s[1];
    }
    out.append(ZWJ + text.slice(at));
    return out;
  }
  function draw() {
    root.replaceChildren();
    data.items.filter((x) => on === "all" || x.lang === on).forEach((x, n) => {
      const [a, b] = x.outputs, nEd = a.edits.length;
      root.append(h("article", { class: "card", style: `animation-delay:${Math.min(n, 10) * 30}ms` },
        h("div", { class: "card-head" },
          h("span", {}, h("b", {}, NAMES[x.lang] || x.lang), ` · ${nEd} error${nEd > 1 ? "s" : ""} each · CER ${pct(x.cer, 1)} for both`),
          h("span", {}, `${x.dataset} · ${x.license}`)),
        h("div", { class: "row" }, h("span", { class: "who" }, "original"), marked(x.reference, a.edits.concat(b.edits), "ref", x.lang), h("span", { class: "sc" })),
        [a, b].map((o) => h("div", { class: "row" },
          h("span", { class: "who" }, o.system),
          marked(o.text, o.edits, "hyp", x.lang),
          h("span", { class: "sc" },
            h("span", { class: "sc-sem" }, "semCER ", h("b", {}, pct(o.semcer, 2))),
            h("span", { class: "sc-cer" }, "CER ", h("b", {}, pct(x.cer, 2))),
            h("span", { class: "sc-meter" }, h("span", { style: `width:${Math.min(100, (100 * o.semcer) / x.cer)}%` }))))),
      ));
    });
  }
  draw();
}

/* ======================================================================
 * 5. Small things
 * ==================================================================== */
function chrome() {
  $$(".copy").forEach((b) => b.addEventListener("click", async () => {
    const code = b.parentElement.querySelector("code").innerText;
    try { await navigator.clipboard.writeText(code); b.textContent = "Copied"; } catch { b.textContent = "Select and copy"; }
    setTimeout(() => (b.textContent = "Copy"), 1600);
  }));
}

chrome();
priceTag();
agreement();
results();
playground().catch(console.error);
gallery().catch(console.error);
})();
