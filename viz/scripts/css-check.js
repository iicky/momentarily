// Parse-only validity check for the app's stylesheets.
//
// Why this exists: `npm run typecheck` compiles TypeScript and never looks at
// CSS, and the tests don't either. An unclosed block therefore passes every
// gate we have and then fails at REQUEST time, as a 500 on every route — which
// is exactly how a bad merge resolution inside a media query shipped a
// stylesheet that broke the whole app while tsc and 169 tests stayed green.
//
// postcss is already present via Next's own toolchain, so this adds no
// dependency. Parse only: no linting, no opinions about style, just "would the
// bundler choke on this".

const fs = require("node:fs");
const path = require("node:path");
const postcss = require("postcss");

const roots = process.argv.slice(2);
const targets = roots.length > 0 ? roots : [path.join(__dirname, "..", "app")];

function cssFilesUnder(target) {
  const stat = fs.statSync(target);
  if (!stat.isDirectory()) return target.endsWith(".css") ? [target] : [];
  return fs
    .readdirSync(target, { withFileTypes: true })
    .flatMap((e) => cssFilesUnder(path.join(target, e.name)));
}

let failed = false;
const files = targets.flatMap(cssFilesUnder);
for (const file of files) {
  try {
    postcss.parse(fs.readFileSync(file, "utf8"), { from: file });
  } catch (err) {
    const where = err.line ? `${err.line}:${err.column ?? 0}` : "?";
    console.error(`${file}:${where} ${err.reason ?? err.message}`);
    failed = true;
  }
}

if (failed) process.exit(1);
console.log(`css-check: ${files.length} file(s) parse cleanly`);
