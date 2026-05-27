import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

const SAMPLE_REPORT = {
  duration_s: 5400,
  players_tracked: 22,
  formation: "4-3-3",
  events: { PASS: 487, SHOT: 24, CROSS: 31, POSSESSION_CHANGE: 203, GOAL: 2 },
  speed_stats: [
    { player_id: 7,  max_speed_kmh: 33.2, avg_speed_kmh: 8.1, distance_m: 9823, sprint_time_s: 42 },
    { player_id: 11, max_speed_kmh: 31.8, avg_speed_kmh: 7.4, distance_m: 10102, sprint_time_s: 38 },
    { player_id: 9,  max_speed_kmh: 30.4, avg_speed_kmh: 9.2, distance_m: 11234, sprint_time_s: 31 },
    { player_id: 10, max_speed_kmh: 28.9, avg_speed_kmh: 7.8, distance_m: 8967, sprint_time_s: 24 },
    { player_id: 4,  max_speed_kmh: 27.3, avg_speed_kmh: 7.1, distance_m: 9441, sprint_time_s: 19 },
    { player_id: 6,  max_speed_kmh: 26.8, avg_speed_kmh: 6.9, distance_m: 8712, sprint_time_s: 17 },
    { player_id: 3,  max_speed_kmh: 25.1, avg_speed_kmh: 6.4, distance_m: 9101, sprint_time_s: 12 },
    { player_id: 8,  max_speed_kmh: 24.7, avg_speed_kmh: 7.3, distance_m: 10543, sprint_time_s: 11 },
  ],
  tactical_log: [
    { t: 312, action: "PASS to (78m, 42m)", xg: 0.04, xt: 0.182, reason: "Teammate in higher threat zone. Through ball advised." },
    { t: 567, action: "SHOOT", xg: 0.31, xt: 0.441, reason: "High-value shooting position inside penalty box. Shot recommended." },
    { t: 891, action: "CROSS or DRIBBLE inside", xg: 0.06, xt: 0.093, reason: "Wide position limits angle. Cutback or driven cross preferred." },
    { t: 1203, action: "RETAIN POSSESSION", xg: 0.02, xt: 0.031, reason: "Low xT zone. Recycle possession to create better opportunity." },
    { t: 1788, action: "SHOOT", xg: 0.28, xt: 0.388, reason: "Clear shooting lane identified. Immediate shot advised." },
  ]
};

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}'${sec.toString().padStart(2, "0")}`;
}

function MiniPitch({ formation = "4-3-3" }) {
  const lines = formation.split("-").map(Number);
  const W = 240, H = 155;
  const players = [];

  // GK
  players.push({ x: 18, y: H / 2 });

  let xStep = (W - 36) / (lines.length + 1);
  lines.forEach((count, li) => {
    const xPos = 18 + xStep * (li + 1);
    const spacing = H / (count + 1);
    for (let i = 0; i < count; i++) {
      players.push({ x: xPos, y: spacing * (i + 1) });
    }
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      <defs>
        <radialGradient id="pitchGrad" cx="50%" cy="50%">
          <stop offset="0%" stopColor="#1a4a1a" />
          <stop offset="100%" stopColor="#0d2e0d" />
        </radialGradient>
      </defs>
      <rect width={W} height={H} rx="4" fill="url(#pitchGrad)" />
      {/* Stripes */}
      {[0,1,2,3,4,5,6].map(i => (
        <rect key={i} x={i * (W/7)} width={W/7} height={H}
          fill={i % 2 === 0 ? "rgba(255,255,255,0.03)" : "transparent"} />
      ))}
      {/* Lines */}
      <rect x="2" y="2" width={W-4} height={H-4} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" rx="3" />
      <line x1={W/2} y1="2" x2={W/2} y2={H-2} stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
      <circle cx={W/2} cy={H/2} r="18" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
      {/* Penalty areas */}
      <rect x="2" y={H/2 - 26} width="28" height="52" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="1" />
      <rect x={W-30} y={H/2 - 26} width="28" height="52" fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth="1" />
      {/* Players */}
      {players.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="6" fill="#1e90ff" stroke="#fff" strokeWidth="1.5" />
        </g>
      ))}
    </svg>
  );
}

function SpeedBar({ player }) {
  const pct = Math.min((player.max_speed_kmh / 36) * 100, 100);
  const sprintPct = Math.min((player.sprint_time_s / 60) * 100, 100);
  const isElite = player.max_speed_kmh >= 30;

  return (
    <div style={{
      padding: "10px 14px",
      borderRadius: 8,
      background: "rgba(255,255,255,0.04)",
      border: `1px solid ${isElite ? "rgba(30,144,255,0.3)" : "rgba(255,255,255,0.08)"}`,
      transition: "all 0.2s",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontFamily: "monospace", fontSize: 13, color: "#aaa" }}>
          #{player.player_id}
        </span>
        <div style={{ display: "flex", gap: 12 }}>
          <span style={{ fontSize: 11, color: "#666" }}>{player.distance_m.toLocaleString()}m</span>
          <span style={{
            fontWeight: 700, fontSize: 14,
            color: isElite ? "#1e90ff" : "#fff"
          }}>
            {player.max_speed_kmh} km/h
          </span>
        </div>
      </div>
      <div style={{ height: 4, background: "rgba(255,255,255,0.08)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: isElite
            ? "linear-gradient(90deg, #1e90ff, #00bfff)"
            : "linear-gradient(90deg, #3a6f3a, #5ab85a)",
          borderRadius: 2,
          transition: "width 1s cubic-bezier(0.4,0,0.2,1)",
        }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        <span style={{ fontSize: 10, color: "#555" }}>avg {player.avg_speed_kmh} km/h</span>
        <span style={{ fontSize: 10, color: "#ff7043" }}>⚡ sprint {player.sprint_time_s}s</span>
      </div>
    </div>
  );
}

function EventTag({ type, count }) {
  const colors = {
    PASS: "#1e90ff",
    SHOT: "#ff4444",
    CROSS: "#ff9800",
    POSSESSION_CHANGE: "#9c27b0",
    GOAL: "#ffd700",
    TACKLE: "#4caf50",
    INTERCEPTION: "#00bcd4",
  };
  const color = colors[type] || "#888";
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "12px 16px",
      background: `${color}18`,
      border: `1px solid ${color}44`,
      borderRadius: 10,
      minWidth: 80,
    }}>
      <span style={{ fontSize: 22, fontWeight: 800, color }}>{count}</span>
      <span style={{ fontSize: 10, color: "#666", textTransform: "uppercase", letterSpacing: 1, marginTop: 2 }}>
        {type.replace("_", " ")}
      </span>
    </div>
  );
}

function TacticalCard({ entry }) {
  const [open, setOpen] = useState(false);
  const xgColor = entry.xg > 0.2 ? "#ff4444" : entry.xg > 0.1 ? "#ff9800" : "#4caf50";

  return (
    <div
      onClick={() => setOpen(!open)}
      style={{
        padding: "10px 14px",
        borderRadius: 8,
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.07)",
        cursor: "pointer",
        transition: "all 0.2s",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 11, color: "#666" }}>{formatTime(entry.t)}</span>
        <div style={{ display: "flex", gap: 8 }}>
          <span style={{ fontSize: 10, color: xgColor }}>xG {entry.xg.toFixed(2)}</span>
          <span style={{ fontSize: 10, color: "#9c27b0" }}>xT {entry.xt.toFixed(3)}</span>
        </div>
      </div>
      <p style={{ margin: "4px 0 0", fontSize: 12, color: "#1e90ff", fontWeight: 600 }}>
        {entry.action}
      </p>
      {open && (
        <p style={{ margin: "6px 0 0", fontSize: 11, color: "#888", lineHeight: 1.5 }}>
          {entry.reason}
        </p>
      )}
    </div>
  );
}

export default function FootballAIDashboard() {
  const [tab, setTab] = useState("overview");
  const [jobId, setJobId] = useState("");
  const [polling, setPolling] = useState(false);
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(SAMPLE_REPORT);
  const [isDemoMode, setIsDemoMode] = useState(true);
  const fileRef = useRef();

  const tabs = ["overview", "speed", "events", "tactical", "upload"];

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("video", file);
    setIsDemoMode(false);
    setStatus({ status: "uploading" });
    try {
      const res = await fetch(`${API}/analyse`, { method: "POST", body: form });
      const data = await res.json();
      setJobId(data.job_id);
      setStatus({ status: "queued" });
      setPolling(true);
    } catch {
      setStatus({ status: "error", error: "Cannot reach backend. Running in demo mode." });
      setIsDemoMode(true);
    }
  }

  useEffect(() => {
    if (!polling || !jobId) return;
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${API}/status/${jobId}`);
        const data = await res.json();
        setStatus(data);
        if (data.status === "complete") {
          setReport(data.report);
          setPolling(false);
        } else if (data.status === "failed") {
          setPolling(false);
        }
      } catch {
        setPolling(false);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [polling, jobId]);

  const card = (children, style = {}) => (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.09)",
      borderRadius: 12,
      padding: 20,
      ...style,
    }}>
      {children}
    </div>
  );

  const label = (text) => (
    <p style={{ margin: "0 0 12px", fontSize: 11, textTransform: "uppercase",
      letterSpacing: 1.5, color: "#555", fontWeight: 600 }}>
      {text}
    </p>
  );

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0f0a",
      color: "#e8e8e8",
      fontFamily: "'DM Mono', 'Courier New', monospace",
      padding: "0",
    }}>
      {/* Header */}
      <div style={{
        padding: "20px 28px",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(0,0,0,0.4)",
        backdropFilter: "blur(10px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, #1e90ff, #00bfff)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>⚽</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, letterSpacing: 1 }}>FOOTBALL AI</div>
            <div style={{ fontSize: 10, color: "#555", letterSpacing: 2 }}>TACTICAL ANALYSIS SYSTEM</div>
          </div>
        </div>

        {isDemoMode && (
          <div style={{
            padding: "4px 10px", borderRadius: 20,
            background: "rgba(255,152,0,0.15)",
            border: "1px solid rgba(255,152,0,0.3)",
            fontSize: 10, color: "#ff9800", letterSpacing: 1,
          }}>
            DEMO MODE
          </div>
        )}

        {status && !isDemoMode && (
          <div style={{
            padding: "4px 10px", borderRadius: 20,
            background: status.status === "complete"
              ? "rgba(76,175,80,0.15)" : "rgba(30,144,255,0.15)",
            border: `1px solid ${status.status === "complete" ? "rgba(76,175,80,0.3)" : "rgba(30,144,255,0.3)"}`,
            fontSize: 10, letterSpacing: 1,
            color: status.status === "complete" ? "#4caf50" : "#1e90ff",
          }}>
            {status.status?.toUpperCase()}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{
        display: "flex", gap: 0,
        padding: "0 28px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        overflowX: "auto",
      }}>
        {tabs.map(t => (
          <button key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "12px 18px",
              background: "none", border: "none",
              borderBottom: tab === t ? "2px solid #1e90ff" : "2px solid transparent",
              color: tab === t ? "#1e90ff" : "#555",
              fontSize: 11, letterSpacing: 1.5,
              textTransform: "uppercase", cursor: "pointer",
              transition: "all 0.2s", fontFamily: "inherit",
              whiteSpace: "nowrap",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "24px 28px", maxWidth: 1100, margin: "0 auto" }}>

        {/* OVERVIEW TAB */}
        {tab === "overview" && report && (
          <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr 300px" }}>
            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {[
                { label: "Duration", value: formatTime(report.duration_s) },
                { label: "Players", value: report.players_tracked },
                { label: "Formation", value: report.formation },
                { label: "Total Events", value: Object.values(report.events).reduce((a, b) => a + b, 0) },
              ].map(s => (
                <div key={s.label} style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 10, padding: "16px 20px",
                }}>
                  <div style={{ fontSize: 10, color: "#555", letterSpacing: 1.5, textTransform: "uppercase" }}>
                    {s.label}
                  </div>
                  <div style={{ fontSize: 26, fontWeight: 800, marginTop: 4, color: "#fff" }}>
                    {s.value}
                  </div>
                </div>
              ))}
              {/* Top speed */}
              {card(
                <>
                  <div style={{ fontSize: 10, color: "#555", letterSpacing: 1.5, textTransform: "uppercase" }}>
                    Top Speed
                  </div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: "#1e90ff", marginTop: 4 }}>
                    {Math.max(...report.speed_stats.map(s => s.max_speed_kmh))} km/h
                  </div>
                  <div style={{ fontSize: 11, color: "#555", marginTop: 2 }}>
                    #{report.speed_stats.sort((a,b)=>b.max_speed_kmh-a.max_speed_kmh)[0].player_id}
                  </div>
                </>,
                { gridColumn: "span 2" }
              )}
            </div>

            {/* Events */}
            {card(
              <>
                {label("Events Detected")}
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {Object.entries(report.events).map(([k, v]) => (
                    <EventTag key={k} type={k} count={v} />
                  ))}
                </div>
              </>
            )}

            {/* Mini pitch */}
            {card(
              <>
                {label(`Formation — ${report.formation}`)}
                <MiniPitch formation={report.formation} />
              </>,
              { display: "flex", flexDirection: "column" }
            )}
          </div>
        )}

        {/* SPEED TAB */}
        {tab === "speed" && report && (
          <div style={{ display: "grid", gap: 16 }}>
            {card(
              <>
                {label("Player Speed & Distance")}
                <div style={{ display: "grid", gap: 8 }}>
                  {[...report.speed_stats]
                    .sort((a, b) => b.max_speed_kmh - a.max_speed_kmh)
                    .map(p => <SpeedBar key={p.player_id} player={p} />)
                  }
                </div>
                <p style={{ marginTop: 14, fontSize: 11, color: "#444" }}>
                  Blue = sprint speed ≥ 30 km/h · Green = standard · ⚡ = sprint time at &gt;25 km/h
                </p>
              </>
            )}
          </div>
        )}

        {/* EVENTS TAB */}
        {tab === "events" && report && (
          <div style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 1fr" }}>
            {Object.entries(report.events).map(([k, v]) => {
              const colors = {
                PASS: "#1e90ff", SHOT: "#ff4444", CROSS: "#ff9800",
                POSSESSION_CHANGE: "#9c27b0", GOAL: "#ffd700",
              };
              const c = colors[k] || "#888";
              const pct = (v / Math.max(...Object.values(report.events))) * 100;
              return card(
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                    <span style={{ fontSize: 11, color: "#666", letterSpacing: 1, textTransform: "uppercase" }}>
                      {k.replace("_", " ")}
                    </span>
                    <span style={{ fontSize: 22, fontWeight: 800, color: c }}>{v}</span>
                  </div>
                  <div style={{ height: 6, background: "rgba(255,255,255,0.07)", borderRadius: 3 }}>
                    <div style={{
                      height: "100%", width: `${pct}%`,
                      background: c, borderRadius: 3,
                      transition: "width 1.2s ease",
                    }} />
                  </div>
                </>
              , { key: k });
            })}
          </div>
        )}

        {/* TACTICAL TAB */}
        {tab === "tactical" && report && (
          <div style={{ display: "grid", gap: 12 }}>
            {card(
              <>
                {label("AI Coach — Tactical Log")}
                <p style={{ margin: "0 0 16px", fontSize: 12, color: "#555" }}>
                  Click any entry to expand the AI reasoning.
                </p>
                <div style={{ display: "grid", gap: 8 }}>
                  {report.tactical_log.map((entry, i) => (
                    <TacticalCard key={i} entry={entry} />
                  ))}
                </div>
              </>
            )}
            {card(
              <>
                {label("xG & xT Explained")}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  {[
                    { label: "xG — Expected Goals", color: "#ff4444",
                      desc: "Probability (0–1) that a shot from this position results in a goal. Based on distance, angle, and context." },
                    { label: "xT — Expected Threat", color: "#9c27b0",
                      desc: "Value of having the ball at a given pitch position. Increases toward the opponent's goal. Drives pass/dribble decisions." },
                  ].map(m => (
                    <div key={m.label} style={{
                      padding: 14, borderRadius: 8,
                      background: `${m.color}10`,
                      border: `1px solid ${m.color}30`,
                    }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: m.color, marginBottom: 6 }}>{m.label}</div>
                      <div style={{ fontSize: 11, color: "#777", lineHeight: 1.6 }}>{m.desc}</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* UPLOAD TAB */}
        {tab === "upload" && (
          <div style={{ display: "grid", gap: 16, maxWidth: 560 }}>
            {card(
              <>
                {label("Upload Match Video")}
                <div
                  onClick={() => fileRef.current?.click()}
                  style={{
                    border: "2px dashed rgba(30,144,255,0.3)",
                    borderRadius: 10,
                    padding: "40px 20px",
                    textAlign: "center",
                    cursor: "pointer",
                    transition: "all 0.2s",
                    background: "rgba(30,144,255,0.04)",
                  }}
                >
                  <div style={{ fontSize: 36, marginBottom: 12 }}>📹</div>
                  <div style={{ fontSize: 13, color: "#aaa" }}>Click to upload video</div>
                  <div style={{ fontSize: 11, color: "#555", marginTop: 6 }}>MP4, AVI, MKV, MOV</div>
                  <input ref={fileRef} type="file" accept="video/*"
                    style={{ display: "none" }} onChange={handleUpload} />
                </div>

                {status && (
                  <div style={{
                    marginTop: 16, padding: "12px 16px", borderRadius: 8,
                    background: status.status === "complete"
                      ? "rgba(76,175,80,0.1)" : "rgba(30,144,255,0.1)",
                    border: `1px solid ${status.status === "complete"
                      ? "rgba(76,175,80,0.3)" : "rgba(30,144,255,0.3)"}`,
                  }}>
                    <div style={{ fontSize: 12, fontWeight: 600,
                      color: status.status === "complete" ? "#4caf50" : "#1e90ff" }}>
                      {status.status === "complete" ? "✓ Analysis complete" :
                       status.status === "processing" ? "⏳ Processing video..." :
                       status.status === "queued" ? "📋 Queued..." :
                       status.status === "error" ? "⚠ " + (status.error || "Error") :
                       status.status}
                    </div>
                    {jobId && <div style={{ fontSize: 10, color: "#555", marginTop: 4 }}>Job: {jobId}</div>}
                  </div>
                )}
              </>
            )}

            {card(
              <>
                {label("Backend Setup")}
                <div style={{ fontSize: 12, color: "#666", lineHeight: 1.8 }}>
                  <div>1. Install dependencies:</div>
                  <div style={{
                    background: "rgba(0,0,0,0.4)", borderRadius: 6,
                    padding: "8px 12px", margin: "6px 0 12px",
                    fontFamily: "monospace", fontSize: 11, color: "#4caf50",
                  }}>
                    pip install -r requirements.txt
                  </div>
                  <div>2. Start the API server:</div>
                  <div style={{
                    background: "rgba(0,0,0,0.4)", borderRadius: 6,
                    padding: "8px 12px", margin: "6px 0 12px",
                    fontFamily: "monospace", fontSize: 11, color: "#4caf50",
                  }}>
                    uvicorn backend.main:app --port 8000
                  </div>
                  <div>3. Or run the pipeline directly:</div>
                  <div style={{
                    background: "rgba(0,0,0,0.4)", borderRadius: 6,
                    padding: "8px 12px", margin: "6px 0",
                    fontFamily: "monospace", fontSize: 11, color: "#4caf50",
                  }}>
                    python run.py
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}