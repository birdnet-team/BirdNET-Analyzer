/* Version switcher for the published docs.
 *
 * Deployed to the gh-pages site root by .github/workflows/documentation.yml and
 * loaded by every published docs version (via html_js_files, or -D for release
 * tags whose conf.py predates it). Reads versions.json, generated at deploy
 * time from the version directories that actually exist on gh-pages.
 *
 * The dropdown replaces the version line the sphinx_rtd_theme prints under the
 * project name in the sidebar; if that element is missing (another theme, a
 * stripped page) it falls back to a small floating box.
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

  function matches(entry) {
    return (
      entry.path === current ||
      (entry.aliases || []).indexOf(current) !== -1
    );
  }

  function build(versions) {
    var select = document.createElement("select");
    select.setAttribute("aria-label", "Documentation version");
    versions.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v.path;
      opt.textContent = v.name;
      opt.selected = matches(v);
      select.appendChild(opt);
    });
    select.addEventListener("change", function () {
      // Keep the page path; 404.html catches pages a version doesn't have.
      location.href = base + select.value + "/" + rest;
    });
    return select;
  }

  function mount(select) {
    // sphinx_rtd_theme: the version line sits under the project name in the
    // sidebar search area. Reuse that slot so the switcher looks native.
    var search = document.querySelector(".wy-side-nav-search");
    if (!search) return false;

    var slot = search.querySelector(".version");
    if (!slot) {
      slot = document.createElement("div");
      slot.className = "version";
      var home = search.querySelector("a");
      if (home && home.nextSibling) {
        search.insertBefore(slot, home.nextSibling);
      } else {
        search.appendChild(slot);
      }
    }
    slot.textContent = "";
    slot.style.opacity = "1";
    select.style.cssText =
      "max-width:100%;padding:2px 4px;border-radius:3px;" +
      "background:rgba(0,0,0,.2);color:#fcfcfc;font-size:90%;" +
      "border:1px solid rgba(255,255,255,.3);cursor:pointer";
    slot.appendChild(select);
    return true;
  }

  function mountFallback(select) {
    var box = document.createElement("div");
    box.style.cssText =
      "position:fixed;bottom:12px;left:12px;z-index:1000;" +
      "background:#1a1a1a;color:#fcfcfc;border-radius:4px;" +
      "padding:6px 10px;font-size:13px;font-family:sans-serif;" +
      "box-shadow:0 1px 4px rgba(0,0,0,.4)";
    var label = document.createElement("label");
    label.textContent = "Version: ";
    select.style.cssText =
      "background:#1a1a1a;color:#fcfcfc;border:none;font-size:13px";
    label.appendChild(select);
    box.appendChild(label);
    document.body.appendChild(box);
  }

  fetch(base + "versions.json")
    .then(function (r) {
      return r.ok ? r.json() : null;
    })
    .then(function (versions) {
      if (!versions || versions.length < 2) return;
      var select = build(versions);
      if (!mount(select)) mountFallback(select);
    })
    .catch(function () {
      /* no versions.json (e.g. local build): no switcher */
    });
})();
