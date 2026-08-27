/* Version switcher for the published docs.
 *
 * Deployed to the gh-pages site root by .github/workflows/documentation.yml and
 * loaded by every published docs version (via html_js_files, or -D for release
 * tags whose conf.py predates it). Reads versions.json, generated at deploy
 * time from the version directories that actually exist on gh-pages.
 */
(function () {
  "use strict";

  // The site root is wherever this script was served from.
  var script = document.currentScript;
  if (!script || !script.src) return;
  var base = script.src.replace(/switcher\.js.*$/, "");

  // Which version dir is the current page under, and what comes after it?
  // Pages not under the site root (e.g. a local sphinx build opened via
  // file://, which still loads this script from the live site) get no
  // switcher.
  if (location.href.indexOf(base) !== 0) return;
  var m = location.href
    .slice(base.length)
    .match(/^(stable|dev|v[0-9][^/]*)(?:\/(.*))?$/);
  if (!m) return;
  var current = m[1];
  var rest = m[2] || "";

  fetch(base + "versions.json")
    .then(function (r) {
      return r.ok ? r.json() : null;
    })
    .then(function (versions) {
      if (!versions || versions.length < 2) return;

      var box = document.createElement("div");
      box.style.cssText =
        "position:fixed;bottom:12px;left:12px;z-index:1000;" +
        "background:#1a1a1a;color:#fcfcfc;border-radius:4px;" +
        "padding:6px 10px;font-size:13px;font-family:sans-serif;" +
        "box-shadow:0 1px 4px rgba(0,0,0,.4)";

      var label = document.createElement("label");
      label.textContent = "Version: ";

      var select = document.createElement("select");
      select.style.cssText =
        "background:#1a1a1a;color:#fcfcfc;border:none;font-size:13px";
      versions.forEach(function (v) {
        var opt = document.createElement("option");
        opt.value = v.path;
        opt.textContent = v.name;
        opt.selected = v.path === current;
        select.appendChild(opt);
      });
      select.addEventListener("change", function () {
        // Keep the page path; 404.html catches pages a version doesn't have.
        location.href = base + select.value + "/" + rest;
      });

      label.appendChild(select);
      box.appendChild(label);
      document.body.appendChild(box);
    })
    .catch(function () {
      /* no versions.json (e.g. local build): no switcher */
    });
})();
