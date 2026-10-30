import { useEffect, useMemo, useState } from "react";
import { getStats, getStudies, getStudy, postDecision, imageUrl } from "./api";
import "./App.css";

const short = (id) => (id ? id.slice(0, 8) : "");

function readSaved(key) {
  try {
    return localStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function ScoreBadge({ study }) {
  if (study.status !== "done") return <span className="badge muted">{study.status}</span>;
  return (
    <span className={`badge ${study.flagged ? "flag" : "ok"}`}>
      {study.flagged ? "FLAGGED" : "not flagged"}
    </span>
  );
}

// Shows the scan and, optionally, the heatmap on top of it.
// App renders this with key={study_id}, so failure/retry state resets when another study is selected.
function StudyImage({ detail, showHeat, heatOpacity }) {
  const [attempt, setAttempt] = useState(0);
  const [scanFailed, setScanFailed] = useState(false);
  const [heatFailed, setHeatFailed] = useState(false);

  const bust = attempt ? `?retry=${attempt}` : "";
  const scanSrc = detail.scan_url ? imageUrl(detail.scan_url) + bust : null;
  const heatSrc = detail.heatmap_url ? imageUrl(detail.heatmap_url) + bust : null;

  function handleError(markFailed) {
    if (attempt === 0) setAttempt(1); // one automatic retry before showing an error
    else markFailed(true);
  }

  function retry() {
    setScanFailed(false);
    setHeatFailed(false);
    setAttempt((a) => a + 1);
  }

  return (
    <>
      <div className="image-box">
        {scanSrc && !scanFailed ? (
          <img
            src={scanSrc}
            alt="Chest X-ray"
            className="scan"
            onError={() => handleError(setScanFailed)}
          />
        ) : (
          <div className="no-image">
            {scanFailed ? (
              <div>
                <p>The image failed to load.</p>
                <button className="link" onClick={retry}>Retry</button>
              </div>
            ) : (
              "No image available"
            )}
          </div>
        )}
        {/* Never draw a heatmap without its scan underneath */}
        {heatSrc && showHeat && !scanFailed && !heatFailed && (
          <img
            src={heatSrc}
            alt="AI explanation heatmap"
            className="heat"
            style={{ opacity: heatOpacity / 100 }}
            onError={() => handleError(setHeatFailed)}
          />
        )}
      </div>
      {heatFailed && (
        <p className="form-error">
          The AI heatmap could not be loaded, so only the plain scan is shown.{" "}
          <button className="link" onClick={retry}>Retry</button>
        </p>
      )}
    </>
  );
}

export default function App() {
  const [studies, setStudies] = useState([]);
  const [stats, setStats] = useState(null);
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [loadedDetail, setLoadedDetail] = useState(null);
  const [detailError, setDetailError] = useState("");
  const [showHeat, setShowHeat] = useState(true);
  const [heatOpacity, setHeatOpacity] = useState(80);
  const [reviewer, setReviewer] = useState(() => readSaved("reviewer"));
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [reloadTick, setReloadTick] = useState(0);

  // Load the list and stats now, then every 10 seconds, so new studies appear by themselves.
  // Changing reloadTick restarts this effect, which reloads immediately.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [list, st] = await Promise.all([getStudies(), getStats()]);
        if (cancelled) return;
        setStudies(list.studies);
        setStats(st);
        // Pin the first study once, so the selection does not jump when new studies arrive
        setSelectedId((prev) => prev ?? list.studies[0]?.study_id ?? null);
        setLoadError("");
      } catch (e) {
        if (!cancelled) {
          setLoadError(`Cannot reach the API (${e.message}). Is it running on port 8000?`);
        }
      }
    };
    const first = setTimeout(load, 0);
    const timer = setInterval(load, 10000);
    return () => {
      cancelled = true;
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [reloadTick]);

  // Load the details (including the audit history) of the selected study.
  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    getStudy(selectedId)
      .then((d) => {
        if (cancelled) return;
        setLoadedDetail(d);
        setDetailError("");
      })
      .catch((e) => {
        if (!cancelled) setDetailError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  // Only show details that belong to the selected study (avoids a flash of the previous one).
  const detail = loadedDetail && loadedDetail.study_id === selectedId ? loadedDetail : null;

  const visible = useMemo(() => {
    let list = studies;
    if (filter === "flagged") list = list.filter((s) => s.flagged);
    if (filter === "pending")
      list = list.filter((s) => s.status === "done" && s.review_status === "pending");
    const q = query.trim().toLowerCase();
    if (q) list = list.filter((s) => s.patient_id.toLowerCase().startsWith(q));
    return list;
  }, [studies, filter, query]);

  const nextPending = visible.find(
    (s) => s.study_id !== selectedId && s.status === "done" && s.review_status === "pending"
  );

  function selectStudy(id) {
    setSelectedId(id);
    setFormError("");
    setDetailError("");
  }

  async function decide(decision) {
    const name = reviewer.trim();
    if (!name) {
      setFormError("Please enter your name as reviewer first.");
      return;
    }
    setBusy(true);
    setFormError("");
    try {
      await postDecision(detail.study_id, { reviewer: name, decision, comment: comment.trim() || null });
      try {
        localStorage.setItem("reviewer", name);
      } catch {
        /* storage may be unavailable */
      }
      setComment("");
      setLoadedDetail(await getStudy(detail.study_id));
      setReloadTick((t) => t + 1); // refresh the list and the counters right away
    } catch (e) {
      setFormError(`Could not save the decision: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  const canReview = detail && detail.status === "done";
  const alreadyReviewed = detail && detail.review_status !== "pending";

  return (
    <div className="app">
      <div className="banner" role="alert">
        FOR EDUCATION ONLY. NOT FOR CLINICAL USE. Do not use these results for any medical decision.
      </div>

      <header className="topbar">
        <h1>AI-Assisted Chest X-ray Review</h1>
        {stats && (
          <div className="stats">
            <span><b>{stats.analysed}</b> analysed</span>
            <span><b>{stats.flagged}</b> flagged</span>
            <span><b>{stats.pending_review}</b> pending</span>
            <span><b>{stats.approved}</b> approved</span>
            <span><b>{stats.rejected}</b> rejected</span>
          </div>
        )}
      </header>

      {loadError && <div className="error-bar">{loadError}</div>}

      <main className="layout">
        <aside className="list">
          <div className="filters">
            {[["all", "All"], ["flagged", "Flagged"], ["pending", "Pending review"]].map(([key, label]) => (
              <button key={key} className={filter === key ? "active" : ""} onClick={() => setFilter(key)}>
                {label}
              </button>
            ))}
          </div>
          <div className="search-wrap">
            <input
              className="text"
              placeholder="Search patient ID…"
              aria-label="Search patient ID"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div className="list-count">
            Showing {visible.length} of {studies.length}
          </div>
          <ul>
            {visible.map((s) => (
              <li
                key={s.study_id}
                className={s.study_id === selectedId ? "selected" : ""}
                onClick={() => selectStudy(s.study_id)}
              >
                <div className="row1">
                  <span className="pid">{short(s.patient_id)}</span>
                  <span className="score">{s.score != null ? s.score.toFixed(2) : "–"}</span>
                </div>
                <div className="row2">
                  <ScoreBadge study={s} />
                  <span className={`review ${s.review_status}`}>{s.review_status}</span>
                </div>
              </li>
            ))}
            {!visible.length && <li className="empty">No studies match this filter.</li>}
          </ul>
        </aside>

        <section className="viewer">
          {!detail &&
            (detailError ? (
              <p className="form-error">Could not load this study: {detailError}</p>
            ) : (
              <p className="hint">
                {studies.length ? "Loading study…" : "No studies yet. Upload scans to Orthanc."}
              </p>
            ))}
          {detail && (
            <>
              <div className="viewer-head">
                <div>
                  <h2>Patient {short(detail.patient_id)}</h2>
                  <p className="meta">
                    {detail.modality} · sex {detail.patient_sex || "?"} · age {detail.patient_age || "?"} ·
                    model {detail.model_version || "none"}
                  </p>
                </div>
                <ScoreBadge study={detail} />
              </div>

              <div className="viewer-body">
                <div className="image-col">
                  <StudyImage
                    key={detail.study_id}
                    detail={detail}
                    showHeat={showHeat}
                    heatOpacity={heatOpacity}
                  />
                  <div className="controls">
                    <label>
                      <input
                        type="checkbox"
                        checked={showHeat && Boolean(detail.heatmap_url)}
                        disabled={!detail.heatmap_url}
                        onChange={(e) => setShowHeat(e.target.checked)}
                      />{" "}
                      Show AI heatmap
                    </label>
                    <label>
                      Opacity{" "}
                      <input
                        type="range"
                        min="10"
                        max="100"
                        value={heatOpacity}
                        disabled={!detail.heatmap_url}
                        onChange={(e) => setHeatOpacity(Number(e.target.value))}
                      />
                    </label>
                  </div>
                  <p className="note">
                    {detail.heatmap_url
                      ? "Warm colours mark image regions that pushed the AI towards pneumonia. The map is coarse and is not an outline of disease."
                      : "No heatmap: the scan was not flagged, and a heatmap on a normal-scored scan would be misleading."}
                  </p>
                </div>

                <div className="panel-col">
                  <div className="card">
                    <h3>AI confidence score</h3>
                    {detail.score != null ? (
                      <>
                        <div className="score-big">{(detail.score * 100).toFixed(0)}%</div>
                        <div className="bar">
                          <div
                            className={`fill ${detail.flagged ? "flag" : "ok"}`}
                            style={{ width: `${detail.score * 100}%` }}
                          />
                          <div
                            className="threshold"
                            style={{ left: `${detail.threshold * 100}%` }}
                            title="Decision threshold"
                          />
                        </div>
                        <p className="note">
                          Threshold {detail.threshold.toFixed(2)}. This is a risk score from the model, not a
                          calibrated probability of disease.
                        </p>
                      </>
                    ) : (
                      <p className="note">{detail.error || "No AI result for this study."}</p>
                    )}
                  </div>

                  <div className="card">
                    <h3>Radiologist review</h3>
                    <p className="current">
                      Current status: <b className={`review ${detail.review_status}`}>{detail.review_status}</b>
                    </p>
                    {alreadyReviewed && (
                      <p className="note">
                        A new decision is added to the history. Earlier entries are never changed or deleted.
                      </p>
                    )}
                    <input
                      className="text"
                      placeholder="Reviewer name (e.g. Dr. Rao)"
                      aria-label="Reviewer name"
                      value={reviewer}
                      maxLength={80}
                      onChange={(e) => setReviewer(e.target.value)}
                    />
                    <textarea
                      className="text"
                      placeholder="Optional comment"
                      aria-label="Comment"
                      value={comment}
                      maxLength={1000}
                      rows={3}
                      onChange={(e) => setComment(e.target.value)}
                    />
                    {formError && <p className="form-error">{formError}</p>}
                    <div className="buttons">
                      <button className="approve" disabled={!canReview || busy} onClick={() => decide("approved")}>
                        Approve AI result
                      </button>
                      <button className="reject" disabled={!canReview || busy} onClick={() => decide("rejected")}>
                        Reject
                      </button>
                    </div>
                    <button
                      className="link"
                      disabled={!nextPending}
                      onClick={() => nextPending && selectStudy(nextPending.study_id)}
                    >
                      {nextPending ? "Next pending study →" : "No other pending studies"}
                    </button>
                  </div>

                  <div className="card">
                    <h3>Audit history ({detail.audit.length})</h3>
                    {!detail.audit.length && <p className="note">No decisions recorded yet.</p>}
                    <ul className="audit">
                      {[...detail.audit].reverse().map((a) => (
                        <li key={a.id}>
                          <b className={`review ${a.decision}`}>{a.decision}</b> by {a.reviewer}
                          <div className="meta">
                            {new Date(a.created_at).toLocaleString()} · AI score {a.ai_score?.toFixed(2)} ·{" "}
                            {a.model_version}
                          </div>
                          {a.comment && <div className="comment">“{a.comment}”</div>}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}