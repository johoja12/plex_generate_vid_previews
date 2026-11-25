let data = null,
  items = [],
  page = 1,
  pages = 1,
  total = 0,
  tab = "all",
  selected = new Set(),
  searchTimer,
  fetchSequence = 0,
  plexId = "",
  refreshing = false;
const labels = {
  next_episode: "Watch history",
  active_next: "Watching next",
  active_playback: "Currently playing",
  recent_partial: "Partially watched",
  fully_cached: "InfiniDysk cache",
  on_deck: "On Deck",
  multi_user_overlap: "Multiple viewers",
};
function theme(name) {
  document.documentElement.dataset.theme = name;
  $("dark-btn").classList.toggle("selected", name === "dark");
  $("light-btn").classList.toggle("selected", name === "light");
  try {
    localStorage.setItem("plex-preview-theme", name);
  } catch (e) {}
}
try {
  theme(
    localStorage.getItem("plex-preview-theme") === "light" ? "light" : "dark",
  );
} catch (e) {
  theme("dark");
}
function sourceLabel(key) {
  const split = key.indexOf(":");
  return key.startsWith("hub:")
    ? key.slice(split + 1)
    : labels[key.slice(split + 1)] || key.slice(split + 1);
}
function reasonLabel(r) {
  return r.type === "hub" ? r.hub : labels[r.type] || r.type;
}
function visibleReasons(row) {
  return [
    ...new Set(
      (row.priority_reasons || [])
        .filter((r) => r.type !== "multi_user_overlap")
        .map(reasonLabel),
    ),
  ];
}
function chip(text) {
  return `<span class="chip ${["Watching next", "Currently playing"].includes(text) ? "orange" : text === "Watch history" ? "blue" : text === "InfiniDysk cache" ? "green" : ""}">${esc(text)}</span>`;
}
function setTab(value) {
  tab = value;
  page = 1;
  selected.clear();
  ["all", "episode", "movie", "failed", "completed", "library"].forEach((t) =>
    $("tab-" + t).classList.toggle("selected", t === tab),
  );
  $("status-filter").value = "";
  fetchItems();
  if (value === "failed") $("queue").scrollIntoView({ behavior: "smooth" });
}
function searchChanged() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    page = 1;
    fetchItems();
  }, 250);
}
function mediaTitle(row) {
  const filename = (row.file_path || "").split("/").pop();
  return filename && row.title.endsWith(" - " + filename)
    ? row.title.slice(0, -filename.length - 3)
    : row.title;
}
async function fetchItems() {
  const seq = ++fetchSequence;
  const params = new URLSearchParams({ page, limit: 10 });
  const status = $("status-filter").value;
  if (status) params.set("status", status);
  else if (["all", "episode", "movie"].includes(tab)) {
    params.set("pending_only", "true");
  } else if (tab === "failed") {
    params.set("attention_only", "true");
  } else if (tab === "completed") {
    params.set("status", "completed");
  }
  if (tab === "episode" || tab === "movie") params.set("media_type", tab);
  if ($("queue-search").value.trim())
    params.set("search", $("queue-search").value.trim());
  if ($("library-filter").value)
    params.set("library_name", $("library-filter").value);
  if ($("source-filter").value) params.set("origin", $("source-filter").value);
  if ($("priority-only").checked) params.set("show_priority_only", "true");
  if ($("hide-done").checked) params.set("hide_completed", "true");
  const sort = $("sort-filter").value;
  if (sort) {
    const [column, order] = sort.split(":");
    params.set("sort_by", column);
    params.set("sort_order", order);
  }
  try {
    const result = await api("/api/items?" + params);
    if (seq !== fetchSequence) return;
    items = result.items;
    pages = Math.max(1, result.pages);
    total = result.total;
    if (page > pages) {
      page = pages;
      return fetchItems();
    }
    renderRows();
  } catch (e) {
    demo(e.message);
  }
}
function renderRows() {
  $("rows").innerHTML =
    items
      .map(
        (row, i) =>
          `<tr><td><input type="checkbox" aria-label="Select ${esc(mediaTitle(row))}" ${selected.has(row.item_id) ? "checked" : ""} onchange="selectItem(${row.item_id},this.checked)"></td><td><button class="row-link" onclick="showItem(${i})">${esc(mediaTitle(row))}</button><div class="row-detail" title="${esc(row.file_path)}">${row.media_type === "episode" ? "TV episode" : "Movie"} · ${esc(row.library_name)}${row.is_multi_part ? " · Version " + (row.part_index + 1) : ""}</div></td><td>${visibleReasons(row).map(chip).join("") || '<span class="score">—</span>'}</td><td class="score">${number(row.priority_score)}</td><td>${chip(row.status === "missing" ? "Waiting" : row.status.replaceAll("_", " "))}${row.status === "processing" ? ` <span class="score">${row.progress}%</span>` : ""}</td><td><button class="row-more" aria-label="Details for ${esc(mediaTitle(row))}" onclick="showItem(${i})">···</button></td></tr>`,
      )
      .join("") ||
    '<tr><td colspan="6" class="empty">No items match these filters.</td></tr>';
  $("row-count").textContent =
    `Page ${page} of ${pages} · ${number(total)} media items`;
  $("prev").disabled = page <= 1;
  $("next").disabled = page >= pages;
  $("queue-caption").textContent =
    tab === "library"
      ? "Library media and versions"
      : tab === "completed"
        ? "Completed previews"
        : tab === "failed"
          ? "Items that need attention"
          : "Ranked by priority";
  $("bulk-bar").hidden = !selected.size;
  $("selected-count").textContent = selected.size + " selected";
  $("select-all").checked =
    items.length > 0 && items.every((row) => selected.has(row.item_id));
  $("select-all").indeterminate =
    items.some((row) => selected.has(row.item_id)) && !$("select-all").checked;
}
function selectItem(id, checked) {
  checked ? selected.add(id) : selected.delete(id);
  renderRows();
}
function selectAll(checked) {
  items.forEach((row) =>
    checked ? selected.add(row.item_id) : selected.delete(row.item_id),
  );
  renderRows();
}
async function bulkAction(action) {
  await confirmWork(
    "Update selected items",
    `Apply ${action.replaceAll("_", " ")} to ${selected.size} media items?`,
    async () => {
      const result = await api("/api/items/bulk-action", {
        method: "POST",
        body: JSON.stringify({ action, item_ids: [...selected] }),
      });
      selected.clear();
      demo(result.message);
      await refresh();
    },
  );
}
function showItem(index) {
  const row = items[index];
  window.detailItem = row;
  const dates = (x) =>
    x ? new Date(x.endsWith("Z") ? x : x + "Z").toLocaleString() : "—";
  const fields = {
    Library: row.library_name,
    Status: row.status,
    File: row.file_path,
    Symlink: row.symlink_target,
    "Bundle hash": row.bundle_hash,
    "Preview file": row.bif_path,
    Added: dates(row.added_at),
    Attempted: dates(row.last_attempted_at),
    Speed: row.avg_speed,
    Error: row.error_message,
  };
  $("dialog-body").innerHTML =
    `<div class="eyebrow">Media details${row.is_multi_part ? " · Version " + (row.part_index + 1) : ""}</div><h2>${esc(mediaTitle(row))}</h2><div>${visibleReasons(row).map(chip).join("")}</div><dl class="details-grid">${Object.entries(
      fields,
    )
      .map(
        ([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value || "—")}</dd>`,
      )
      .join(
        "",
      )}</dl><div class="item-actions"><button onclick="itemAction('move','top')">Move to top</button><button onclick="itemAction('move','bottom')">Move to bottom</button><button onclick="itemAction('move','reset')">Requeue</button><button onclick="itemAction('regenerate')">Regenerate</button><button onclick="itemAction('analyze')">Plex analyze</button><button onclick="itemAction('reset-to-missing')">Reset</button><button onclick="itemAction('mark-completed')">Mark completed</button><button onclick="itemAction('mark-failed')">Mark failed</button><button onclick="itemAction('force-hash-calc')">Calculate hash</button><button onclick="itemAction('log-debug-json')">Log debug JSON</button>${plexId ? `<a class="chip" target="_blank" rel="noopener" href="https://app.plex.tv/desktop/#!/server/${encodeURIComponent(plexId)}/details?key=${encodeURIComponent("/library/metadata/" + row.item_id)}">Open in Plex ↗</a>` : ""}</div><div id="details-output"></div><div class="dialog-actions"><button onclick="$('app-dialog').close()">Close</button></div>`;
  $("app-dialog").showModal();
}
async function itemAction(action, position) {
  const row = window.detailItem;
  $("app-dialog").close();
  const confirmNeeded = !["log-debug-json"].includes(action);
  const work = async () => {
    const result = await api(`/api/items/${row.item_id}/${action}`, {
      method: "POST",
      ...(position ? { body: JSON.stringify({ position }) } : {}),
    });
    demo(result.message);
    await refresh();
  };
  if (confirmNeeded)
    await confirmWork(
      "Update media",
      `${action === "move" ? "Move to " + position : action.replaceAll("-", " ")}: ${mediaTitle(row)}?${action === "regenerate" ? " Existing preview files will be regenerated." : ""}`,
      work,
    );
  else
    try {
      await work();
    } catch (e) {
      demo(e.message);
    }
}
async function togglePause() {
  const button = $("pause");
  button.disabled = true;
  try {
    const result = await api(
      "/api/queue/" + (data.settings.queue_paused ? "resume" : "pause"),
      { method: "POST" },
    );
    demo(result.message);
    await refresh();
  } catch (e) {
    demo(e.message);
  } finally {
    button.disabled = false;
  }
}
async function syncLibrary() {
  try {
    const result = await api("/api/sync", { method: "POST" });
    demo(result.message);
  } catch (e) {
    demo(e.message);
  }
}
function renderOverview() {
  const c = data.counts,
    total = Object.values(c).reduce((a, b) => a + b, 0);
  $("hour").textContent = number(data.completed_hour);
  $("completed").textContent = number(c.COMPLETED);
  $("library-count").textContent = `of ${number(total)} library items`;
  $("pending").textContent = number(data.pending);
  $("failed").textContent = number((c.FAILED || 0) + (c.SLOW_FAILED || 0));
  $("scope-label").textContent = data.settings.full_backlog
    ? "Full backlog"
    : "Priority only";
  const maximum = Math.max(1, ...data.hour_bins);
  $("chart").innerHTML = data.hour_bins
    .map(
      (n, i) =>
        `<i style="height:${Math.max(3, (n / maximum) * 28)}px" title="${n} unique completions · interval ${i + 1}"></i>`,
    )
    .join("");
  $("pause").textContent = data.settings.queue_paused
    ? "▶ Resume queue"
    : "Ⅱ Pause queue";
  $("live-state").textContent = data.settings.queue_paused
    ? "● Queue paused"
    : "● Live";
  $("processing").innerHTML = data.processing.length
    ? data.processing
        .map(
          (r) =>
            `<div class="processing-card"><div class="poster"><svg width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.3"><use href="#play"/></svg></div><div class="process-info"><div class="item-title">${esc(r.title)}</div><div class="process-detail">${esc(r.library_name)} · ${visibleReasons(r).map(esc).join(" + ") || "Preview generation"}</div><div class="progress-track"><div class="progress-fill" style="width:${Math.max(0, Math.min(100, r.progress))}%"></div></div><div class="processing-path" title="${esc(r.symlink_target || r.file_path)}">${esc(r.symlink_target || r.file_path || "")}</div></div><div class="process-stats"><span class="chip green">${esc(r.processing_worker_type || "Worker")}</span><strong>${r.progress}%</strong><span style="color:var(--muted)">${r.current_fps ? Number(r.current_fps).toFixed(1) + " fps" : "Starting…"}</span><span style="color:var(--green)">${esc(r.avg_speed || "—")}</span></div></div>`,
        )
        .join("")
    : '<div class="empty">No previews are processing right now.</div>';
  const entries = Object.entries(data.sources).sort((a, b) => b[1] - a[1]);
  const highest = Math.max(1, ...entries.map((e) => e[1]));
  $("sources").innerHTML =
    entries
      .map(
        ([source, count]) =>
          `<div class="source"><div class="source-line"><span>${esc(sourceLabel(source))}</span><strong>${count}</strong></div><div class="source-bar"><i style="width:${(count / highest) * 100}%"></i></div></div>`,
      )
      .join("") +
    '<div class="source-note">Items may have multiple origins.<br>TV hubs select 20 entries and share a limit of 3 episodes per show in 24 hours.</div>';
  $("activity").innerHTML = data.recent_completed.length
    ? data.recent_completed
        .slice(0, 5)
        .map(
          (r) =>
            `<div class="activity-row"><span class="check">✓</span><div><div class="activity-title">${esc(r.title)}</div><div class="activity-time">Preview completed · ${elapsed(r.at)}</div></div></div>`,
        )
        .join("")
    : '<div class="empty">No generated items completed in the last hour.</div>';
  const old = $("source-filter").value;
  $("source-filter").innerHTML =
    '<option value="">All sources</option>' +
    entries
      .map(([s]) => `<option value="${esc(s)}">${esc(sourceLabel(s))}</option>`)
      .join("");
  if (entries.some(([s]) => s === old)) $("source-filter").value = old;
  $("freshness").textContent =
    "Live · updated " +
    new Date(data.snapshot_at).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  $("sync-time").textContent =
    "Library synced " + elapsed(data.settings.last_sync_time);
  const rate = data.rate_limit;
  $("rate-limit").textContent = rate?.is_limited
    ? rate.message
    : rate?.limit_gb
      ? `${number(rate.remaining_bytes / 1024 ** 3)} GB remaining this hour`
      : "";
}
async function fetchPlayback() {
  try {
    const result = await api("/api/playback");
    $("watching").innerHTML =
      result.available === false
        ? "Playback information is temporarily unavailable."
        : result.sessions.length
          ? result.sessions
              .map(
                (s) =>
                  `<div class="playback-card"><strong>${esc(s.title)}</strong><p>${esc(s.viewer || s.username || "Viewer")} · ${esc(s.device || s.player || "Plex")} · ${s.duration ? Math.round((s.view_offset / s.duration) * 100) : 0}% · ${esc(s.preview_status || "Preview status unavailable")}</p></div>`,
              )
              .join("")
          : "No active Plex playback.";
  } catch (e) {
    $("watching").textContent =
      "Playback information is temporarily unavailable.";
  }
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const [overview, processing, stats] = await Promise.all([
      api("/api/overview"),
      api("/api/items/processing"),
      api("/api/stats"),
    ]);
    data = {
      ...overview,
      processing: processing.items.filter((r) => r.status === "processing"),
      rate_limit: stats.rate_limit,
    };
    renderOverview();
    await fetchItems();
  } catch (e) {
    console.error(e);
    $("freshness").textContent = "Live data unavailable";
    demo(e.message);
  } finally {
    refreshing = false;
  }
}
async function init() {
  try {
    const [libs, settings] = await Promise.all([
      api("/api/libraries"),
      api("/api/settings"),
    ]);
    $("library-filter").innerHTML =
      '<option value="">All libraries</option>' +
      libs.map((l) => `<option value="${esc(l)}">${esc(l)}</option>`).join("");
    plexId = settings.plex_client_identifier || "";
  } catch (e) {
    demo(e.message);
  }
  if (new URLSearchParams(location.search).get("view") === "library") {
    tab = "library";
    $("tab-all").classList.remove("selected");
    $("tab-library").classList.add("selected");
  }
  await refresh();
  fetchPlayback();
  setInterval(refresh, 5000);
  setInterval(fetchPlayback, 15000);
}
init();
