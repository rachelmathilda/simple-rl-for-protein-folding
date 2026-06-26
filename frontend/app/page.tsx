"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────

interface FoldResult {
  sequence: string;
  energy: number;
  optimal_energy: number | null;
  positions: [number, number][];
  contacts: [number, number][];
  n_rollouts: number;
  energy_history: number[];
}

interface Benchmark {
  sequence: string;
  length: number;
  optimal_energy: number;
}

// ─────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:7860";

const CELL   = 52;   // px per grid cell
const R_H    = 20;   // radius hydrophobic node
const R_P    = 16;   // radius polar node
const PAD    = 48;   // canvas padding

// ─────────────────────────────────────────────
// COLORS
// ─────────────────────────────────────────────

const C = {
  bg          : "#0b0e14",
  panel       : "#111520",
  border      : "#1e2535",
  accent      : "#5b7fff",
  accentDim   : "#2a3a7a",
  hydrophobic : "#e05c5c",
  polar       : "#5cb8e0",
  contact     : "#f0a03a",
  backbone    : "#334155",
  text        : "#e2e8f0",
  textMuted   : "#64748b",
  textDim     : "#334155",
  success     : "#34d399",
  warning     : "#fbbf24",
  mono        : "var(--font-mono)",
};

// ─────────────────────────────────────────────
// HELPER — draw folding onto canvas
// ─────────────────────────────────────────────

function drawFolding(
  canvas: HTMLCanvasElement,
  result: FoldResult,
  highlight: number | null
) {
  const ctx  = canvas.getContext("2d");
  if (!ctx) return;

  const { sequence, positions, contacts } = result;
  const n = positions.length;
  if (n === 0) return;

  const minX = Math.min(...positions.map((p) => p[0]));
  const minY = Math.min(...positions.map((p) => p[1]));
  const maxX = Math.max(...positions.map((p) => p[0]));
  const maxY = Math.max(...positions.map((p) => p[1]));

  const w = (maxX - minX) * CELL + PAD * 2 + R_H * 2;
  const h = (maxY - minY) * CELL + PAD * 2 + R_H * 2;

  canvas.width  = Math.max(w, 300);
  canvas.height = Math.max(h, 300);

  // Flip Y: SVG-style top-left origin
  const px = (x: number) => (x - minX) * CELL + PAD + R_H;
  const py = (y: number) => canvas.height - ((y - minY) * CELL + PAD + R_H);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grid dots
  ctx.fillStyle = C.textDim;
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      ctx.beginPath();
      ctx.arc(px(x), py(y), 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // H-H contact lines
  for (const [i, j] of contacts) {
    const [xi, yi] = positions[i];
    const [xj, yj] = positions[j];
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = C.contact;
    ctx.lineWidth   = 1.5;
    ctx.globalAlpha = 0.6;
    ctx.moveTo(px(xi), py(yi));
    ctx.lineTo(px(xj), py(yj));
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
  }

  // Backbone
  ctx.beginPath();
  ctx.strokeStyle = C.backbone;
  ctx.lineWidth   = 3;
  ctx.lineJoin    = "round";
  for (let i = 0; i < n; i++) {
    const [x, y] = positions[i];
    i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y));
  }
  ctx.stroke();

  // Nodes
  for (let i = 0; i < n; i++) {
    const [x, y] = positions[i];
    const isH    = sequence[i] === "H";
    const r      = isH ? R_H : R_P;
    const isHL   = highlight === i;

    // Outer glow on hover
    if (isHL) {
      ctx.beginPath();
      ctx.arc(px(x), py(y), r + 7, 0, Math.PI * 2);
      ctx.fillStyle = isH ? C.hydrophobic : C.polar;
      ctx.globalAlpha = 0.2;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Fill
    ctx.beginPath();
    ctx.arc(px(x), py(y), r, 0, Math.PI * 2);
    const grad = ctx.createRadialGradient(
      px(x) - r * 0.3, py(y) - r * 0.3, r * 0.1,
      px(x), py(y), r
    );
    const baseColor = isH ? C.hydrophobic : C.polar;
    grad.addColorStop(0, isHL ? "#ffffff" : lighten(baseColor, 0.3));
    grad.addColorStop(1, baseColor);
    ctx.fillStyle = grad;
    ctx.fill();

    // Border
    ctx.strokeStyle = isHL ? "#ffffff" : darken(baseColor, 0.3);
    ctx.lineWidth   = isHL ? 2.5 : 1.5;
    ctx.stroke();

    // Label
    ctx.fillStyle  = "#ffffff";
    ctx.font       = `bold ${r * 0.7}px var(--font-mono, monospace)`;
    ctx.textAlign  = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(sequence[i], px(x), py(y));
  }
}

function lighten(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, (n >> 16) + Math.round(255 * amt));
  const g = Math.min(255, ((n >> 8) & 0xff) + Math.round(255 * amt));
  const b = Math.min(255, (n & 0xff) + Math.round(255 * amt));
  return `rgb(${r},${g},${b})`;
}
function darken(hex: string, amt: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.max(0, (n >> 16) - Math.round(255 * amt));
  const g = Math.max(0, ((n >> 8) & 0xff) - Math.round(255 * amt));
  const b = Math.max(0, (n & 0xff) - Math.round(255 * amt));
  return `rgb(${r},${g},${b})`;
}

// ─────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────

function SequenceTag({ char, index, highlight, onHover }: {
  char: string;
  index: number;
  highlight: number | null;
  onHover: (i: number | null) => void;
}) {
  const isH  = char === "H";
  const isHL = highlight === index;
  return (
    <span
      onMouseEnter={() => onHover(index)}
      onMouseLeave={() => onHover(null)}
      style={{
        display      : "inline-flex",
        alignItems   : "center",
        justifyContent:"center",
        width        : "26px",
        height       : "26px",
        borderRadius : "5px",
        fontSize     : "11px",
        fontFamily   : C.mono,
        fontWeight   : "700",
        cursor       : "default",
        color        : "#fff",
        background   : isHL
          ? isH ? C.hydrophobic : C.polar
          : isH ? C.accentDim : "#1e3a4a",
        border       : `1.5px solid ${isH ? C.hydrophobic : C.polar}`,
        opacity      : isHL ? 1 : 0.7,
        transition   : "all 0.12s ease",
        userSelect   : "none",
      }}
    >
      {char}
    </span>
  );
}

function Stat({ label, value, accent }: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:"3px" }}>
      <span style={{ fontSize:"11px", color: C.textMuted, textTransform:"uppercase", letterSpacing:"0.08em" }}>
        {label}
      </span>
      <span style={{
        fontSize  : "22px",
        fontWeight: "700",
        fontFamily: C.mono,
        color     : accent ? C.accent : C.text,
      }}>
        {value}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────
// MAIN PAGE
// ─────────────────────────────────────────────

export default function Home() {
  const [sequence,   setSequence]   = useState("HPHPPHHPHP");
  const [rollouts,   setRollouts]   = useState(64);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState<string | null>(null);
  const [result,     setResult]     = useState<FoldResult | null>(null);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [highlight,  setHighlight]  = useState<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch benchmarks on mount
  useEffect(() => {
    fetch(`${API_URL}/benchmarks`)
      .then((r) => r.json())
      .then((d) => setBenchmarks(d.sequences ?? []))
      .catch(() => {});
  }, []);

  // Redraw when result or highlight changes
  useEffect(() => {
    if (!result || !canvasRef.current) return;
    drawFolding(canvasRef.current, result, highlight);
  }, [result, highlight]);

  const handleFold = useCallback(async () => {
    const seq = sequence.trim().toUpperCase();
    if (!seq || !/^[HP]+$/.test(seq)) {
      setError("Sequence must contain only H and P characters.");
      return;
    }
    if (seq.length < 6 || seq.length > 64) {
      setError("Sequence length must be between 6 and 64.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/fold`, {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ sequence: seq, n_rollouts: rollouts }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? `HTTP ${res.status}`);
      }

      const data: FoldResult = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [sequence, rollouts]);

  const approxRatio =
    result && result.optimal_energy
      ? (result.energy / result.optimal_energy).toFixed(3)
      : null;

  const isOptimal =
    result &&
    result.optimal_energy !== null &&
    result.energy === result.optimal_energy;

  // ── RENDER ──────────────────────────────────

  return (
    <div style={{
      minHeight      : "100vh",
      background     : C.bg,
      color          : C.text,
      fontFamily     : "var(--font-inter), system-ui, sans-serif",
      display        : "flex",
      flexDirection  : "column",
    }}>

      {/* Header */}
      <header style={{
        padding        : "16px 32px",
        borderBottom   : `1px solid ${C.border}`,
        display        : "flex",
        alignItems     : "center",
        gap            : "12px",
        background     : C.panel,
      }}>
        <div style={{
          width        : "32px",
          height       : "32px",
          borderRadius : "8px",
          background   : `linear-gradient(135deg, ${C.accent}, ${C.hydrophobic})`,
          flexShrink   : 0,
        }} />
        <div>
          <div style={{ fontWeight: "700", fontSize: "16px", letterSpacing: "-0.02em" }}>
            HP Fold
          </div>
          <div style={{ fontSize: "12px", color: C.textMuted }}>
            PPO-based HP lattice protein folding
          </div>
        </div>
        <div style={{ marginLeft:"auto", display:"flex", gap:"8px", alignItems:"center" }}>
          <div style={{
            fontSize    : "11px",
            fontFamily  : C.mono,
            color       : C.textMuted,
            padding     : "4px 10px",
            border      : `1px solid ${C.border}`,
            borderRadius: "6px",
          }}>
            PPO agent
          </div>
        </div>
      </header>

      {/* Main two-panel layout */}
      <main style={{
        flex          : 1,
        display       : "grid",
        gridTemplateColumns: "380px 1fr",
        gap           : "0",
        overflow      : "hidden",
      }}>

        {/* LEFT: Input panel */}
        <aside style={{
          background    : C.panel,
          borderRight   : `1px solid ${C.border}`,
          padding       : "28px 24px",
          display       : "flex",
          flexDirection : "column",
          gap           : "24px",
          overflowY     : "auto",
        }}>

          {/* Sequence input */}
          <section>
            <label style={{
              display     : "block",
              fontSize    : "12px",
              fontWeight  : "600",
              color       : C.textMuted,
              marginBottom: "8px",
              textTransform:"uppercase",
              letterSpacing:"0.08em",
            }}>
              HP Sequence
            </label>
            <textarea
              value={sequence}
              onChange={(e) => setSequence(e.target.value.toUpperCase())}
              rows={3}
              spellCheck={false}
              placeholder="e.g. HPHPPHHPHP"
              style={{
                width           : "100%",
                background      : C.bg,
                border          : `1.5px solid ${C.border}`,
                borderRadius    : "8px",
                color           : C.text,
                fontFamily      : C.mono,
                fontSize        : "15px",
                letterSpacing   : "0.06em",
                padding         : "12px 14px",
                resize          : "vertical",
                outline         : "none",
                boxSizing       : "border-box",
                lineHeight      : "1.6",
              }}
              onFocus={(e) => (e.target.style.borderColor = C.accent)}
              onBlur={(e)  => (e.target.style.borderColor = C.border)}
            />
            <div style={{
              marginTop  : "6px",
              fontSize   : "11px",
              color      : C.textMuted,
              display    : "flex",
              justifyContent:"space-between",
            }}>
              <span>Only H (hydrophobic) and P (polar) characters</span>
              <span style={{ fontFamily: C.mono }}>
                {sequence.replace(/[^HP]/gi,"").length} / 64
              </span>
            </div>
          </section>

          {/* Rollouts */}
          <section>
            <label style={{
              display     : "block",
              fontSize    : "12px",
              fontWeight  : "600",
              color       : C.textMuted,
              marginBottom: "8px",
              textTransform:"uppercase",
              letterSpacing:"0.08em",
            }}>
              Rollouts
            </label>
            <div style={{ display:"flex", gap:"8px", alignItems:"center" }}>
              <input
                type="range"
                min={1}
                max={256}
                value={rollouts}
                onChange={(e) => setRollouts(Number(e.target.value))}
                style={{ flex:1, accentColor: C.accent }}
              />
              <span style={{
                fontFamily : C.mono,
                fontSize   : "14px",
                color      : C.text,
                minWidth   : "36px",
                textAlign  : "right",
              }}>
                {rollouts}
              </span>
            </div>
            <div style={{ fontSize:"11px", color: C.textMuted, marginTop:"4px" }}>
              More rollouts = better chance of finding the optimal conformation.
            </div>
          </section>

          {/* Run button */}
          <button
            onClick={handleFold}
            disabled={loading}
            style={{
              padding        : "13px",
              background     : loading ? C.accentDim : C.accent,
              color          : "#fff",
              border         : "none",
              borderRadius   : "8px",
              fontSize       : "14px",
              fontWeight     : "600",
              cursor         : loading ? "not-allowed" : "pointer",
              transition     : "background 0.15s",
              letterSpacing  : "0.02em",
            }}
          >
            {loading ? "Folding…" : "Fold Sequence"}
          </button>

          {/* Error */}
          {error && (
            <div style={{
              padding     : "12px 14px",
              background  : "#2d1515",
              border      : "1px solid #7f2020",
              borderRadius: "8px",
              fontSize    : "13px",
              color       : "#f87171",
              lineHeight  : "1.5",
            }}>
              {error}
            </div>
          )}

          {/* Results stats */}
          {result && (
            <section style={{
              background  : C.bg,
              border      : `1px solid ${C.border}`,
              borderRadius: "10px",
              padding     : "20px",
              display     : "grid",
              gridTemplateColumns:"1fr 1fr",
              gap         : "20px",
            }}>
              <Stat label="Energy Found" value={result.energy} />
              <Stat
                label="Optimal Energy"
                value={result.optimal_energy ?? "unknown"}
                accent
              />
              {approxRatio && (
                <Stat label="Approx Ratio" value={approxRatio} />
              )}
              <Stat label="Chain Length" value={result.sequence.length} />
              <Stat label="H-H Contacts" value={result.contacts.length} />
              <Stat label="Rollouts" value={result.n_rollouts} />

              {isOptimal && (
                <div style={{
                  gridColumn  : "1 / -1",
                  display     : "flex",
                  alignItems  : "center",
                  gap         : "8px",
                  padding     : "10px 12px",
                  background  : "#0d2e20",
                  borderRadius: "7px",
                  border      : `1px solid ${C.success}`,
                  fontSize    : "13px",
                  color       : C.success,
                  fontWeight  : "600",
                }}>
                  <span>Optimal conformation found</span>
                </div>
              )}
            </section>
          )}

          {/* Sequence residue list (hoverable) */}
          {result && (
            <section>
              <div style={{
                fontSize    : "12px",
                color       : C.textMuted,
                marginBottom: "10px",
                textTransform:"uppercase",
                letterSpacing:"0.08em",
                fontWeight  : "600",
              }}>
                Residues
              </div>
              <div style={{
                display   : "flex",
                flexWrap  : "wrap",
                gap       : "4px",
              }}>
                {result.sequence.split("").map((char, i) => (
                  <SequenceTag
                    key={i}
                    char={char}
                    index={i}
                    highlight={highlight}
                    onHover={setHighlight}
                  />
                ))}
              </div>
              <div style={{ fontSize:"11px", color: C.textMuted, marginTop:"8px" }}>
                Hover a residue to highlight it on the lattice.
              </div>
            </section>
          )}

          {/* Benchmark picker */}
          {benchmarks.length > 0 && (
            <section>
              <div style={{
                fontSize    : "12px",
                color       : C.textMuted,
                marginBottom: "10px",
                textTransform:"uppercase",
                letterSpacing:"0.08em",
                fontWeight  : "600",
              }}>
                Benchmark Sequences
              </div>
              <div style={{ display:"flex", flexDirection:"column", gap:"6px" }}>
                {benchmarks.map((b) => (
                  <button
                    key={b.sequence}
                    onClick={() => setSequence(b.sequence)}
                    style={{
                      background   : sequence === b.sequence ? C.accentDim : "transparent",
                      border       : `1px solid ${sequence === b.sequence ? C.accent : C.border}`,
                      borderRadius : "7px",
                      padding      : "8px 12px",
                      color        : C.text,
                      cursor       : "pointer",
                      textAlign    : "left",
                      display      : "flex",
                      justifyContent:"space-between",
                      alignItems   : "center",
                      fontSize     : "12px",
                      transition   : "all 0.1s",
                    }}
                  >
                    <span style={{ fontFamily: C.mono, fontSize:"11px", color: C.textMuted }}>
                      N={b.length}
                    </span>
                    <span style={{
                      fontFamily : C.mono,
                      fontSize   : "11px",
                      maxWidth   : "200px",
                      overflow   : "hidden",
                      textOverflow:"ellipsis",
                      whiteSpace : "nowrap",
                    }}>
                      {b.sequence}
                    </span>
                    <span style={{ color: C.accent, fontFamily: C.mono, fontSize:"11px" }}>
                      E*={b.optimal_energy}
                    </span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </aside>

        {/* RIGHT: Visualization panel */}
        <section style={{
          display       : "flex",
          flexDirection : "column",
          overflow      : "hidden",
          position      : "relative",
        }}>

          {/* Legend bar */}
          <div style={{
            padding        : "12px 24px",
            borderBottom   : `1px solid ${C.border}`,
            display        : "flex",
            alignItems     : "center",
            gap            : "24px",
            background     : C.panel,
            flexShrink     : 0,
          }}>
            {[
              { color: C.hydrophobic, label: "H — Hydrophobic" },
              { color: C.polar,       label: "P — Polar" },
              { color: C.contact,     label: "H-H contact" },
              { color: C.backbone,    label: "Backbone" },
            ].map(({ color, label }) => (
              <div key={label} style={{ display:"flex", alignItems:"center", gap:"7px" }}>
                <div style={{
                  width       : "10px",
                  height      : "10px",
                  borderRadius: "50%",
                  background  : color,
                  flexShrink  : 0,
                }} />
                <span style={{ fontSize:"12px", color: C.textMuted }}>{label}</span>
              </div>
            ))}

            {result && (
              <div style={{ marginLeft:"auto", fontFamily: C.mono, fontSize:"12px", color: C.textMuted }}>
                {result.sequence.length} residues
              </div>
            )}
          </div>

          {/* Canvas area */}
          <div style={{
            flex       : 1,
            overflow   : "auto",
            display    : "flex",
            alignItems : "flex-start",
            justifyContent:"flex-start",
            padding    : "24px",
          }}>
            {!result && !loading && (
              <div style={{
                margin    : "auto",
                textAlign : "center",
                color     : C.textMuted,
              }}>
                <div style={{
                  width        : "72px",
                  height       : "72px",
                  borderRadius : "50%",
                  border       : `2px dashed ${C.border}`,
                  margin       : "0 auto 16px",
                  display      : "flex",
                  alignItems   : "center",
                  justifyContent:"center",
                  fontSize     : "28px",
                }}>
                  ⬡
                </div>
                <div style={{ fontSize:"16px", fontWeight:"600", color: C.text, marginBottom:"8px" }}>
                  No conformation yet
                </div>
                <div style={{ fontSize:"13px", lineHeight:"1.6" }}>
                  Enter an HP sequence on the left<br />and click Fold Sequence.
                </div>
              </div>
            )}

            {loading && (
              <div style={{
                margin    : "auto",
                textAlign : "center",
                color     : C.textMuted,
              }}>
                <div style={{
                  width        : "48px",
                  height       : "48px",
                  border       : `3px solid ${C.accentDim}`,
                  borderTop    : `3px solid ${C.accent}`,
                  borderRadius : "50%",
                  margin       : "0 auto 16px",
                  animation    : "spin 0.8s linear infinite",
                }} />
                <div style={{ fontSize:"14px", color: C.text }}>Running PPO rollouts…</div>
                <div style={{ fontSize:"12px", marginTop:"6px" }}>{rollouts} rollouts</div>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </div>
            )}

            {result && !loading && (
              <canvas
                ref={canvasRef}
                style={{
                  display    : "block",
                  borderRadius:"12px",
                  background : "#0d1117",
                  boxShadow  : `0 0 0 1px ${C.border}`,
                  imageRendering:"pixelated",
                }}
              />
            )}
          </div>

          {/* Energy history mini-chart */}
          {result && result.energy_history.length > 1 && (
            <div style={{
              borderTop   : `1px solid ${C.border}`,
              padding     : "16px 24px",
              background  : C.panel,
              flexShrink  : 0,
            }}>
              <div style={{
                fontSize    : "11px",
                color       : C.textMuted,
                marginBottom: "10px",
                textTransform:"uppercase",
                letterSpacing:"0.08em",
                fontWeight  : "600",
              }}>
                Energy across {result.n_rollouts} rollouts
              </div>
              <EnergyMiniChart
                values={result.energy_history}
                best={result.energy}
                optimal={result.optimal_energy}
              />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────
// MINI CHART — energy history as sparkline SVG
// ─────────────────────────────────────────────

function EnergyMiniChart({ values, best, optimal }: {
  values  : number[];
  best    : number;
  optimal : number | null;
}) {
  const W = 600;
  const H = 60;
  const pad = 4;

  const minE = Math.min(...values);
  const maxE = Math.max(...values, 0);
  const range = maxE - minE || 1;

  const toX = (i: number) => pad + (i / (values.length - 1)) * (W - pad * 2);
  const toY = (v: number) => H - pad - ((v - minE) / range) * (H - pad * 2);

  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      style={{ display:"block" }}
    >
      {/* Optimal energy line */}
      {optimal !== null && (
        <line
          x1={pad} y1={toY(optimal)}
          x2={W - pad} y2={toY(optimal)}
          stroke={C.accent}
          strokeWidth={1}
          strokeDasharray="4 3"
          opacity={0.5}
        />
      )}

      {/* Energy path */}
      <path
        d={path}
        fill="none"
        stroke={C.contact}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.8}
      />

      {/* Best energy dot */}
      {(() => {
        const bestIdx = values.indexOf(best);
        if (bestIdx === -1) return null;
        return (
          <circle
            cx={toX(bestIdx)}
            cy={toY(best)}
            r={4}
            fill={C.success}
          />
        );
      })()}

      {/* Labels */}
      <text x={W - pad} y={toY(minE) - 4} textAnchor="end"
        fill={C.textMuted} fontSize="10" fontFamily="monospace">
        best={minE}
      </text>
      {optimal !== null && (
        <text x={W - pad} y={toY(optimal) - 4} textAnchor="end"
          fill={C.accent} fontSize="10" fontFamily="monospace">
          E*={optimal}
        </text>
      )}
    </svg>
  );
}
