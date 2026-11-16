(() => {
  const storageKey = "cra-review-page-draft";
  const base = window.CRA_CONTENT;
  const fields = {
    title: "Paper title",
    repository: "GitHub notice",
    results: "Experiment Results heading",
    restaurant: "Restaurant section",
    restaurantNote: "Restaurant video caption",
    speed: "High-speed section",
    hole: "Warehouse hole section",
    aggregation: "Hot-pot aggregation section",
    aggregationNote: "Hot-pot video caption",
    abstractTitle: "Abstract heading",
    abstract: "Abstract",
    pipeline: "Pipeline heading"
  };

  function draftContent() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      const valid = Object.entries(saved || {}).filter(([key, value]) =>
        Object.hasOwn(fields, key) && typeof value === "string");
      return { ...base, ...Object.fromEntries(valid) };
    } catch {
      return { ...base };
    }
  }

  const editor = document.getElementById("edit-form");
  if (!editor) {
    const preview = new URLSearchParams(location.search).get("draft") === "1";
    const content = preview ? draftContent() : base;
    const titlePrefix = "CRA:";
    document.title = content.title;
    for (const node of document.querySelectorAll("[data-copy]")) {
      const copy = content[node.dataset.copy] ?? "";
      if (node.classList.contains("paper-title") && copy.startsWith(titlePrefix + " ")) {
        const acronym = document.createElement("span");
        acronym.className = "paper-acronym";
        acronym.textContent = titlePrefix;
        node.replaceChildren(acronym, copy.slice(titlePrefix.length));
      } else {
        node.textContent = copy;
      }
    }
    return;
  }

  const editable = draftContent();
  const fieldGroup = document.getElementById("edit-fields");
  const status = document.getElementById("edit-status");
  for (const [name, labelText] of Object.entries(fields)) {
    const row = document.createElement("div");
    row.className = "row";
    const label = document.createElement("label");
    label.htmlFor = "field-" + name;
    label.textContent = labelText;
    const input = document.createElement(name === "abstract" ? "textarea" : "input");
    input.name = name;
    input.id = label.htmlFor;
    input.value = editable[name] ?? "";
    input.required = true;
    row.append(label, input);
    fieldGroup.append(row);
  }

  const editedValues = () => Object.fromEntries(new FormData(editor));
  editor.addEventListener("submit", (event) => {
    event.preventDefault();
    try {
      localStorage.setItem(storageKey, JSON.stringify(editedValues()));
      location.href = "index.html?draft=1";
    } catch {
      status.textContent = "The browser could not save this draft. You can still download the edited content.";
    }
  });
  document.getElementById("download-content").addEventListener("click", () => {
    if (!editor.reportValidity()) return;
    const data = "window.CRA_CONTENT = " + JSON.stringify(editedValues(), null, 2) + ";\n";
    const url = URL.createObjectURL(new Blob([data], { type: "text/javascript" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "content.js";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
    status.textContent = "Downloaded content.js.";
  });
})();
