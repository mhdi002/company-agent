// Render a proposal JSON spec to .docx with docx-js.
// Follows /mnt/skills/public/docx/SKILL.md: built-in HeadingLevel for TOC, numbering config for bullets
// (never literal "•"), tables with columnWidths AND per-cell DXA widths, ShadingType.CLEAR, PageBreak inside
// a Paragraph, no "\n" inside runs, page size set explicitly.
// Usage: node tools/docx_render.js spec.json out.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, PageBreak, TableOfContents, Footer, Header, PageNumber, LevelFormat,
} = require("docx");

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = process.argv[3];
const A4 = { width: 11906, height: 16838 };
const LETTER = { width: 12240, height: 15840 };
const page = spec.page_size === "Letter" ? LETTER : A4;
const MARGIN = 1440;
const CONTENT = page.width - 2 * MARGIN;
const ACCENT = "1F4E79";

const lines = (s) => String(s ?? "").split(/\n+/).filter((x) => x.trim().length);
const para = (text, opts = {}) => lines(text).map((t) => new Paragraph({ spacing: { after: 120 }, ...opts,
  children: [new TextRun({ text: t, ...(opts.run || {}) })] }));

function table(tbl) {
  const n = tbl.headers.length;
  // First column narrower when it is a short index/label column.
  const first = n > 2 ? Math.round(CONTENT * 0.08) : Math.round(CONTENT * 0.3);
  const rest = n > 1 ? Math.floor((CONTENT - first) / (n - 1)) : CONTENT;
  const widths = [first, ...Array(Math.max(n - 1, 0)).fill(rest)];
  widths[n - 1] += CONTENT - widths.reduce((a, b) => a + b, 0); // widths must sum to the table width
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, header) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: header ? { type: ShadingType.CLEAR, color: "auto", fill: "DCE6F1" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: lines(text).length ? lines(text).map((t) => new Paragraph({ children: [new TextRun({ text: t, bold: header })] }))
      : [new Paragraph({ children: [] })],
  });
  return new Table({
    width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: tbl.headers.map((h, i) => cell(h, i, true)) }),
      ...tbl.rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}

const children = [];
const cover = spec.sections.find((s) => s.title === "Cover");
const [title, ...sub] = cover ? cover.paragraphs : [spec.title];
children.push(new Paragraph({ spacing: { before: 3600, after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: title, bold: true, size: 52, color: ACCENT })] }));
for (const s of sub) children.push(...para(s, { alignment: AlignmentType.CENTER, run: { size: 24, color: "404040" } }));
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Table of Contents")] }));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

for (const s of spec.sections) {
  if (s.title === "Cover") continue;
  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(s.title)] }));
  for (const p of s.paragraphs || []) children.push(...para(p));
  for (const b of s.bullets || []) children.push(...para(b, { numbering: { reference: "bullets", level: 0 } }));
  if (s.table && s.table.rows && s.table.rows.length) {
    children.push(table(s.table));
    children.push(new Paragraph({ children: [] }));
  }
}

const doc = new Document({
  creator: "ProposalAgent", title: spec.title, description: "Automatically researched project proposal",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: ACCENT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: page, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
      titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: spec.title, size: 16, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 18 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => fs.writeFileSync(out, buf)).catch((e) => { console.error(e); process.exit(1); });
