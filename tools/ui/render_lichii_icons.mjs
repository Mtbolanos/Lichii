// Renders the Lichii theme's SVG sources to the PNGs the stylesheet and dialogs load,
// at 1x and @2x, and lists them in style.qrc. Qt picks the @2x file on high-DPI screens.
// Menu icons (menu_*) also get a white _white variant for the highlighted menu row.
// Needs @resvg/resvg-js:
//   npm i @resvg/resvg-js && node tools/ui/render_lichii_icons.mjs
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { Resvg } from "@resvg/resvg-js";

const root = new URL("../../dist/qt_themes/lichii/", import.meta.url);
const src = new URL("svg/", root);
const out = new URL("icons/", root);
mkdirSync(out, { recursive: true });

const render = (svg, name) => {
    const width = Number(/width="(\d+)"/.exec(svg)[1]);
    for (const [suffix, scale] of [["", 1], ["@2x", 2]]) {
        const png = new Resvg(svg, { fitTo: { mode: "width", value: width * scale } }).render().asPng();
        writeFileSync(new URL(name + suffix + ".png", out), png);
    }
};

for (const file of readdirSync(src).filter((f) => f.endsWith(".svg")).sort()) {
    const svg = readFileSync(new URL(file, src), "utf8");
    const name = file.replace(".svg", "");
    render(svg, name);
    if (name.startsWith("menu_")) {
        render(svg.replace(/#[0-9a-f]{6}/gi, "#ffffff"), name + "_white");
    }
}

const qrc = new URL("style.qrc", root);
const icons = readdirSync(out).filter((f) => f.endsWith(".png")).sort();
const list = icons.map((f) => `        <file>icons/${f}</file>\n`).join("");
const text = readFileSync(qrc, "utf8").replace(
    /(        <file>style\.qss<\/file>\n)(?:        <file>icons\/[^<]+<\/file>\n)*/,
    (_, qss) => qss + list,
);
writeFileSync(qrc, text);
