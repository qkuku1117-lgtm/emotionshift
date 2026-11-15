// Builds deliverables/05_anqi/slides.pptx in the team's deck style (dark title slide, serif titles, light cards).
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");
const THEME = { name: "EmotionShift", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "0B0B0B", lt1: "FCFCFB", dk2: "17202F", lt2: "EEF1F7", accent1: "2A6FC9", accent2: "EB6834", accent3: "1BAF7A",
            accent4: "EDA100", accent5: "6250D6", accent6: "5A6270", hlink: "2A6FC9", folHlink: "6250D6" } };
const NOTES = [
"Hi, I'm Anqi. My part is B4: RoBERTa with context. B3 read one sentence at a time. My question is: what changes when the model can also read the conversation before it?",
"It is the same RoBERTa as B3, with the same training recipe. Only the input changes. Before the target sentence we add the previous k turns, each with its speaker's name. If the input gets too long, we cut the oldest turns, so the target is always kept. I chose k on the dev set.",
"This is the core. For every turn, it takes a window of the last k turns and joins them as speaker, colon, text. With k equal to zero it returns just the sentence, which is exactly B3's input. So the comparison with B3 is fair.",
"On the test sets, context helps on all three datasets: 63.3 on MELD, 58.7 on DailyDialog and 84.4 on EmoWOZ. That is half a point, three points and six points above B3. And ten turns or the whole history give the same score.",
"The key finding is where the gain comes from. Stable turns, where the emotion stays the same, gain three, seven and eleven points. Shift turns gain almost nothing on MELD and DailyDialog. So context helps the model continue an emotion, not notice a change.",
"Does it really use the conversation? If we shuffle in turns from another dialogue, the score drops below having no history at all. Shuffling the speaker names changes nothing. And with class weights, B4-plus beats B3 on macro-F1 by three points.",
"" ];
(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "B4 RoBERTa with context"; pres.author = "HU ANQI";
  const C = pres.SchemeColor;
  const FOOT = "Workstream #5 · RoBERTa with context";
  pres.defineSlideMaster({ title: "TITLE", background: { color: C.text2 }, objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.75, w: 8.8, h: 1.0, fontSize: 44, bold: true, color: C.background1, margin: 0, valign: "middle", align: "left" }, text: "" } }] });
  pres.defineSlideMaster({ title: "CONTENT", background: { color: C.background1 }, objects: [
    { text: { text: FOOT, options: { x: 0.6, y: 5.1, w: 6, h: 0.3, fontSize: 11, color: "5A6270", margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 8.8, h: 0.75, fontSize: 30, bold: true, color: C.text2, margin: 0, valign: "middle", align: "left" }, text: "" } }],
    slideNumber: { x: 8.9, y: 5.1, w: 0.5, h: 0.3, fontSize: 11, color: "5A6270", align: "right" } });
  pres.addSection({ title: "B4" });
  const add = (master) => pres.addSlide({ masterName: master, sectionTitle: "B4" });
  const kicker = (s, text, x, y, w, name, color) => s.addText(text.toUpperCase(), { x, y, w, h: 0.25, fontSize: 10.5, bold: true, color: color || C.accent1, charSpacing: 2, margin: 0, isTextBox: true, objectName: name });
  const cardBox = (s, x, y, w, h, name) => s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { type: "none" }, rectRadius: 0.1, objectName: name });

  // 1. title
  let s = add("TITLE");
  kicker(s, "EmotionShift · Workstream #5", 0.6, 1.3, 8, "Kicker", "8FB8EE");
  s.addText("RoBERTa with context", { placeholder: "title" });
  s.addText("What changes when the model can also read the conversation before the utterance?", { x: 0.6, y: 3.0, w: 7.6, h: 0.8, fontSize: 18, color: "C9D1DC", margin: 0, valign: "top", isTextBox: true, objectName: "Question" });
  s.addText("HU ANQI · EE6405 Group Project, Table 4", { x: 0.6, y: 4.7, w: 8, h: 0.3, fontSize: 12, color: "8FB8EE", margin: 0, isTextBox: true, objectName: "Byline" });
  s.addNotes(NOTES[0]);

  // 2. how it works
  s = add("CONTENT");
  s.addText("How B4 works", { placeholder: "title" });
  const inputCard = (x, w, label, lines, name) => {
    cardBox(s, x, 1.3, w, 1.95, name + " card");
    kicker(s, label, x + 0.25, 1.45, w - 0.5, name + " label");
    s.addText(lines.map((t, i) => ({ text: t.text, options: { color: t.dim ? "5A6270" : "17202F", bold: !t.dim, breakLine: i < lines.length - 1 } })),
      { x: x + 0.25, y: 1.85, w: w - 0.5, h: 1.25, fontSize: 14, fontFace: "Courier New", margin: 0, valign: "top", isTextBox: true, objectName: name + " input" });
  };
  inputCard(0.6, 3.5, "B3 input · k = 0", [{ text: "It is fine, really!" }], "B3");
  inputCard(4.3, 5.1, "B4 input · k = 2", [{ text: "Rachel: I did not.", dim: true }, { text: "Joey: Oh. I am sorry.", dim: true }, { text: "Rachel: It is fine, really!" }], "B4");
  const steps = [["Same model, same recipe", "RoBERTa-base, as B3; 5 seeds"], ["Only the input changes", "previous k turns with speaker names"], ["Long input", "oldest turns are cut, the target stays"], ["Window k", "chosen on dev only"]];
  steps.forEach(([h, t], i) => {
    const x = 0.6 + i * 2.2;
    s.addText([{ text: h, options: { bold: true, color: C.text2, fontSize: 14, breakLine: true } }, { text: t, options: { color: C.text1, fontSize: 12 } }],
      { x, y: 3.6, w: 2.05, h: 1.0, margin: 0, valign: "top", isTextBox: true, objectName: "Step " + (i + 1) });
  });
  s.addNotes(NOTES[1]);

  // 3. core in code
  s = add("CONTENT");
  s.addText("The core in code", { placeholder: "title" });
  cardBox(s, 0.6, 1.3, 8.8, 2.35, "Code card");
  const code = ["def dialogue_inputs(turns, k):", "    out = []", "    for j in range(len(turns)):", "        window = (turns[: j + 1] if k is None", "                  else turns[max(0, j - k): j + 1])",
                "        out.append(window[-1][1] if k == 0 else", "                   \"\\n\".join(f\"{s}: {t}\" for s, t in window))", "    return out"];
  s.addText(code.map((t, i) => ({ text: t, options: { breakLine: i < code.length - 1 } })),
    { x: 0.85, y: 1.45, w: 8.3, h: 2.05, fontSize: 13, fontFace: "Courier New", color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: "Code" });
  s.addText("The last k turns → one input, target last", { x: 0.6, y: 3.85, w: 8.8, h: 0.35, fontSize: 18, bold: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "Code takeaway" });
  s.addText("k = 0 returns the sentence alone, which is B3's input, so B4 − B3 measures context only.", { x: 0.6, y: 4.25, w: 8.8, h: 0.3, fontSize: 14, color: C.text1, margin: 0, isTextBox: true, objectName: "Code note" });
  s.addText("Source: emotionshift/data/context.py · lines 40–46 (dialogue_inputs; line breaks added)", { x: 0.6, y: 4.7, w: 8.8, h: 0.25, fontSize: 10.5, color: "5A6270", margin: 0, isTextBox: true, objectName: "Code source" });
  s.addNotes(NOTES[2]);

  // 4. test results
  s = add("CONTENT");
  s.addText("Test results", { placeholder: "title" });
  [["MELD · weighted-F1", "63.3", "+0.5 over B3 · 5 seeds"], ["DailyDialog · micro-F1, no neutral", "58.7", "+3.4 over B3 · test_dedup"], ["EmoWOZ · weighted-F1, no neutral", "84.4", "+6.3 over B3 · 5 seeds"]].forEach(([k, big, sub], i) => {
    const x = 0.6 + i * 3.0;
    cardBox(s, x, 1.25, 2.8, 1.55, "Result card " + (i + 1));
    kicker(s, k, x + 0.2, 1.38, 2.5, "Result label " + (i + 1));
    s.addText(big, { x: x + 0.2, y: 1.66, w: 2.4, h: 0.7, fontSize: 40, bold: true, color: C.text2, fontFace: "Cambria", margin: 0, valign: "middle", isTextBox: true, objectName: "Result value " + (i + 1) });
    s.addText(sub, { x: x + 0.2, y: 2.4, w: 2.5, h: 0.28, fontSize: 11.5, color: "5A6270", margin: 0, isTextBox: true, objectName: "Result note " + (i + 1) });
  });
  const th = (t, al) => ({ text: t, options: { bold: true, color: "17202F", align: al || "right", fontSize: 12.5 } });
  const td = (t, al, hi) => ({ text: t, options: { color: "0B0B0B", align: al || "right", fontSize: 12.5, bold: !!hi, fill: hi ? { color: "DCE8F8" } : undefined } });
  s.addTable([[th("Test score (%)", "left"), th("MELD"), th("DailyDialog"), th("EmoWOZ")],
      [td("B3 RoBERTa, no context", "left"), td("62.9"), td("55.3"), td("78.1")],
      [td("B4, last 10 turns (mine)", "left", true), td("63.3", "right", true), td("58.7", "right", true), td("84.4", "right", true)],
      [td("B4, whole history", "left"), td("63.3"), td("59.6"), td("84.3")]],
    { x: 0.6, y: 3.05, w: 8.8, colW: [3.7, 1.7, 1.7, 1.7], rowH: 0.36, border: { type: "solid", color: "C9D1DC", pt: 0.75 }, objectName: "Results table" });
  s.addText("10 turns or the whole history: no difference (every 95% interval includes 0).", { x: 0.6, y: 4.6, w: 8.8, h: 0.28, fontSize: 12, italic: true, color: "5A6270", margin: 0, isTextBox: true, objectName: "Window note" });
  s.addNotes(NOTES[3]);

  // 5. key finding
  s = add("CONTENT");
  s.addText("Context helps most on stable turns", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
      { name: "Stable turns", labels: ["MELD", "DailyDialog", "EmoWOZ"], values: [3.12, 7.38, 11.15] },
      { name: "Shift turns", labels: ["MELD", "DailyDialog", "EmoWOZ"], values: [-1.05, 0.73, 4.33] }],
    { x: 0.6, y: 1.2, w: 5.5, h: 3.75, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60, chartColors: ["2A6FC9", "EB6834"],
      showTitle: true, title: "B4 − B3, points", titleFontSize: 12, titleColor: "5A6270", titleFontFace: "+mn-lt",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0", dataLabelFontSize: 11, dataLabelColor: "0B0B0B", dataLabelFontFace: "+mn-lt",
      catAxisLabelColor: "5A6270", valAxisLabelColor: "5A6270", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
      valGridLine: { color: "E1E4EA", size: 0.75 }, catGridLine: { style: "none" }, valAxisMinVal: -3, valAxisMaxVal: 12, valAxisMajorUnit: 3,
      showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: "0B0B0B", legendFontFace: "+mn-lt", objectName: "Gain by stratum chart" });
  [["Stable turns", "+3.1, +7.4, +11.2 points. Every interval excludes 0.", C.accent1], ["Shift turns", "MELD −1.1, DailyDialog +0.7 (not significant), EmoWOZ +4.3.", C.accent2],
   ["So", "Context helps the model continue an emotion, not notice a change.", C.text2]].forEach(([h, t, col], i) => {
    const y = 1.35 + i * 1.15;
    s.addShape(pres.ShapeType.ellipse, { x: 6.4, y: y + 0.04, w: 0.3, h: 0.3, fill: { color: col }, line: { type: "none" }, objectName: "Finding marker " + (i + 1) });
    s.addText([{ text: h, options: { bold: true, color: C.text2, fontSize: 16, breakLine: true } }, { text: t, options: { color: C.text1, fontSize: 13.5 } }],
      { x: 6.85, y, w: 2.55, h: 1.05, margin: 0, valign: "top", isTextBox: true, objectName: "Finding " + (i + 1) });
  });
  s.addNotes(NOTES[4]);

  // 6. controls and B4+
  s = add("CONTENT");
  s.addText("It uses the right conversation", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "DailyDialog, micro-F1 excl. neutral (%)", labels: ["Real history", "No history", "Shuffled history", "Shuffled speakers"], values: [59.47, 51.86, 50.06, 59.42] }],
    { x: 0.6, y: 1.2, w: 5.2, h: 3.3, barDir: "col", barGapWidthPct: 70, chartColors: ["2A6FC9"],
      showTitle: true, title: "DailyDialog, micro-F1 excl. neutral (%)", titleFontSize: 12, titleColor: "5A6270", titleFontFace: "+mn-lt",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 11, dataLabelColor: "0B0B0B", dataLabelFontFace: "+mn-lt",
      catAxisLabelColor: "5A6270", valAxisLabelColor: "5A6270", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
      valGridLine: { color: "E1E4EA", size: 0.75 }, catGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 70, valAxisMajorUnit: 10, showLegend: false, objectName: "Controls chart" });
  s.addText("Turns from another dialogue are worse than no history; speaker names change nothing.", { x: 0.6, y: 4.5, w: 5.2, h: 0.5, fontSize: 12, italic: true, color: "5A6270", margin: 0, valign: "top", isTextBox: true, objectName: "Controls note" });
  cardBox(s, 6.1, 1.3, 3.3, 2.2, "B4 plus card");
  kicker(s, "B4+ · class weights, MELD", 6.3, 1.45, 3.0, "B4 plus label");
  s.addText("+3.2", { x: 6.3, y: 1.75, w: 2.9, h: 0.8, fontSize: 40, bold: true, color: C.text2, fontFace: "Cambria", margin: 0, valign: "middle", isTextBox: true, objectName: "B4 plus gain" });
  s.addText("macro-F1 over B3 (40.6 → 43.8); the 95% interval excludes 0", { x: 6.3, y: 2.6, w: 2.9, h: 0.7, fontSize: 12.5, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "B4 plus note" });
  s.addText("Limits: gains on shift turns stay small; that is the gap ShiftAware targets.", { x: 6.1, y: 3.75, w: 3.3, h: 0.9, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Limits" });
  s.addNotes(NOTES[5]);

  // 7. references
  s = add("CONTENT");
  s.addText("References", { placeholder: "title" });
  const refs = [
    "Feng, S., Lubis, N., Geishauser, C., Lin, H.-C., Heck, M., van Niekerk, C., & Gašić, M. (2022). EmoWOZ: A large-scale corpus and labelling scheme for emotion recognition in task-oriented dialogue systems. Proceedings of LREC 2022, 4096-4113.",
    "Kim, T., & Vossen, P. (2021). EmoBERTa: Speaker-aware emotion recognition in conversation with RoBERTa. arXiv preprint arXiv:2108.12009.",
    "Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). DailyDialog: A manually labelled multi-turn dialogue dataset. Proceedings of IJCNLP 2017, 986-995.",
    "Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.",
    "Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). MELD: A multimodal multi-party dataset for emotion recognition in conversations. Proceedings of ACL 2019, 527-536.",
    "Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. Proceedings of EMNLP 2020: System Demonstrations, 38-45." ];
  s.addText(refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 8 } })),
    { x: 0.6, y: 1.25, w: 8.8, h: 3.7, fontSize: 12, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Reference list" });

  await pres.writeFile({ fileName: process.argv[2] });
  await applyTheme(process.argv[2], THEME);
})();
