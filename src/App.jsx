import { useState, useEffect, useCallback, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';

const API = 'https://web-production-81316.up.railway.app';

const DIST_NORMAL     = 300;
const DIST_PRECAUCION = 200;
const DIST_ALERTA     = 100;
const DIST_EMERGENCIA =  50;

function getEstado(distCm) {
  if (distCm >= 900) return { l: 'SIN LECTURA', c: '#64748b', bg: '#1e293b', bc: '#334155' };
  if (distCm <= DIST_EMERGENCIA) return { l: 'EMERGENCIA', c: '#ef4444', bg: '#2d0a0a', bc: '#7f1d1d' };
  if (distCm <= DIST_ALERTA)     return { l: 'ALERTA',     c: '#f97316', bg: '#2d1500', bc: '#7c2d12' };
  if (distCm <= DIST_PRECAUCION) return { l: 'PRECAUCIÓN', c: '#f59e0b', bg: '#2d2000', bc: '#78350f' };
  return { l: 'NORMAL', c: '#22c55e', bg: '#0f2d1a', bc: '#166534' };
}

const RIESGO_CFG = {
  'CRITICO': { c: '#ef4444', bg: '#2d0a0a' },
  'ALTO':    { c: '#f97316', bg: '#2d1500' },
  'MEDIO':   { c: '#f59e0b', bg: '#2d2000' },
  'BAJO':    { c: '#22c55e', bg: '#0f2d1a' },
  'MINIMO':  { c: '#3b82f6', bg: '#1e3a5f' },
};

const IMP_LABELS = {
  dist_actual:   'Distancia actual',
  tasa_cambio:   'Tasa de cambio',
  aceleracion:   'Aceleración',
  dist_min:      'Distancia mínima',
  diff_promedio: 'Diff. promedio',
  voltaje:       'Voltaje batería',
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>
      <div style={{ color: '#64748b', marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => p.value !== null && (
        <div key={i} style={{ color: p.color || '#60a5fa', fontWeight: 600 }}>{p.name}: {parseFloat(p.value).toFixed(1)} cm</div>
      ))}
    </div>
  );
}

function KpiCard({ accent, label, value, valueSub, color, sub }) {
  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: '14px 16px', position: 'relative', overflow: 'hidden' }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 3, background: accent, borderRadius: '10px 10px 0 0' }} />
      <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 8, fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: valueSub ? 18 : 26, fontWeight: 700, lineHeight: 1, marginBottom: 4, color: color || '#f1f5f9' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b' }}>{sub}</div>
    </div>
  );
}

// ── Panel ML Random Forest ─────────────────────────────────────
function PanelML() {
  const [pred, setPred] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchPred = useCallback(async () => {
    try {
      const res = await fetch(`${API}/prediccion`);
      const data = await res.json();
      if (!data.error) setPred(data);
    } catch {}
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchPred();
    const t = setInterval(fetchPred, 30000);
    return () => clearInterval(t);
  }, [fetchPred]);

  if (loading) return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>Cargando modelo ML...</div>
    </div>
  );
  if (!pred) return null;

  const cfg = RIESGO_CFG[pred.riesgo] || RIESGO_CFG['MINIMO'];
  const score = pred.riesgo_score || 0;
  const tasa = pred.tasa_cambio_cm_min || 0;
  const eta = pred.minutos_a_precaucion;
  const etaColor = eta !== null && eta < 30 ? '#ef4444' : eta !== null && eta < 60 ? '#f59e0b' : '#22c55e';
  const tasaColor = tasa < -0.5 ? '#ef4444' : tasa > 0.5 ? '#22c55e' : '#94a3b8';
  const impEntries = pred.importancia_features ? Object.entries(pred.importancia_features).sort((a, b) => b[1] - a[1]) : [];
  const maxImp = impEntries.length ? impEntries[0][1] : 1;

  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '.07em' }}>
          Predicción ML — Random Forest (scikit-learn)
        </span>
        <div style={{ display: 'flex', gap: 8 }}>
          <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: pred.modelo_entrenado ? '#0f2d1a' : '#2d2000', color: pred.modelo_entrenado ? '#22c55e' : '#f59e0b', border: `1px solid ${pred.modelo_entrenado ? '#166534' : '#78350f'}`, fontWeight: 500 }}>
            {pred.modelo_entrenado ? 'RF entrenado' : 'Modo lineal'}
          </span>
          <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: '#1e3a5f', color: '#60a5fa', border: '1px solid #1d4ed8', fontWeight: 500 }}>
            {pred.n_muestras} muestras
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
        <div style={{ background: '#0f172a', borderRadius: 10, padding: 14, textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 8 }}>Score de riesgo</div>
          <div style={{ fontFamily: 'monospace', fontSize: 52, fontWeight: 700, color: cfg.c, lineHeight: 1 }}>{score}</div>
          <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>/100</div>
          <div style={{ display: 'inline-block', fontSize: 12, fontWeight: 700, padding: '4px 16px', borderRadius: 6, marginTop: 8, background: cfg.bg, color: cfg.c, letterSpacing: '.05em' }}>{pred.riesgo}</div>
          <div style={{ marginTop: 12, height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: score + '%', background: cfg.c, borderRadius: 4, transition: 'width .8s ease' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#334155', marginTop: 3 }}>
            <span>MÍN</span><span>BAJO</span><span>MEDIO</span><span>ALTO</span><span>CRÍT</span>
          </div>
        </div>

        <div style={{ background: '#0f172a', borderRadius: 10, padding: 14, textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 8 }}>Tiempo a precaución</div>
          <div style={{ fontFamily: 'monospace', fontSize: 52, fontWeight: 700, color: etaColor, lineHeight: 1 }}>{eta !== null ? eta : 'N/A'}</div>
          <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>minutos</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 14 }}>
            <div style={{ background: '#1e293b', borderRadius: 6, padding: 8 }}>
              <div style={{ fontSize: 9, color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>Distancia</div>
              <div style={{ fontSize: 14, fontWeight: 600, fontFamily: 'monospace', color: '#60a5fa' }}>{(pred.dist_actual_cm || 0).toFixed(1)} cm</div>
            </div>
            <div style={{ background: '#1e293b', borderRadius: 6, padding: 8 }}>
              <div style={{ fontSize: 9, color: '#475569', textTransform: 'uppercase', marginBottom: 4 }}>Tasa</div>
              <div style={{ fontSize: 14, fontWeight: 600, fontFamily: 'monospace', color: tasaColor }}>{tasa.toFixed(2)}</div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ background: '#0f172a', borderLeft: '3px solid #3b82f6', borderRadius: '0 8px 8px 0', padding: '12px 14px', fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginBottom: 14 }}>
        {pred.interpretacion}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: impEntries.length ? 14 : 0 }}>
        {[
          { label: 'Distancia actual', val: (pred.dist_actual_cm||0).toFixed(1)+' cm', pct: Math.min(100,((pred.dist_actual_cm||0)/DIST_NORMAL)*100), color: '#3b82f6' },
          { label: 'Tasa de cambio', val: tasa.toFixed(3)+' cm/min', pct: Math.min(100,Math.abs(tasa)*10), color: tasa<0?'#ef4444':'#22c55e' },
          { label: 'Aceleración', val: (pred.aceleracion||0).toFixed(4), pct: Math.min(100,Math.abs(pred.aceleracion||0)*5), color: '#f59e0b' },
          { label: 'Dist. mín. reciente', val: (pred.dist_min_cm||0).toFixed(1)+' cm', pct: Math.min(100,((pred.dist_min_cm||0)/DIST_NORMAL)*100), color: '#ef4444' },
        ].map(f => (
          <div key={f.label} style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, padding: '10px 12px' }}>
            <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>{f.label}</div>
            <div style={{ fontSize: 15, fontWeight: 600, fontFamily: 'monospace', color: f.color }}>{f.val}</div>
            <div style={{ height: 4, background: '#1e293b', borderRadius: 2, marginTop: 6, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: f.pct + '%', background: f.color, borderRadius: 2 }} />
            </div>
          </div>
        ))}
      </div>

      {impEntries.length > 0 && (
        <div>
          <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 10 }}>Importancia de variables — Random Forest</div>
          {impEntries.map(([k, v]) => (
            <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 7 }}>
              <span style={{ fontSize: 11, color: '#94a3b8', width: 120, flexShrink: 0 }}>{IMP_LABELS[k] || k}</span>
              <div style={{ flex: 1, height: 8, background: '#0f172a', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: Math.round((v / maxImp) * 100) + '%', background: '#3b82f6', borderRadius: 4 }} />
              </div>
              <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#60a5fa', width: 40, textAlign: 'right' }}>{(v * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#334155', marginTop: 12, borderTop: '1px solid #1e293b', paddingTop: 10 }}>
        <span>{pred.metodo}</span>
        <span>{pred.total_mediciones_db} registros en DB</span>
      </div>
    </div>
  );
}

// ── Panel Predicción + Proyección ─────────────────────────────
function PanelPrediccionProyeccion() {
  const [datos, setDatos] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDatos = useCallback(async () => {
    try {
      const [rHist, rProy, rLSTM] = await Promise.all([
        fetch(`${API}/historial?limite=20`),
        fetch(`${API}/proyeccion`),
        fetch(`${API}/prediccion-lstm`)
      ]);
      const hist = (await rHist.json()).reverse().filter(m => m.nivel_cm < 900);
      const proy = await rProy.json();
      const lstm = await rLSTM.json();
      setDatos({ hist, proy, lstm });
    } catch {}
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchDatos();
    const t = setInterval(fetchDatos, 30000);
    return () => clearInterval(t);
  }, [fetchDatos]);

  if (loading) return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>Cargando predicción y proyección...</div>
    </div>
  );
  if (!datos || datos.proy.error) return null;

  const { hist, proy, lstm } = datos;
  const tasa    = proy.tasa_cm_min || 0;
  const lstmEta = lstm.disponible ? lstm.minutos_a_precaucion : null;
  const etaColor = lstmEta !== null && lstmEta < 60 ? '#ef4444' : lstmEta !== null && lstmEta < 240 ? '#f59e0b' : '#3b82f6';

  // Construir datos de la gráfica combinada
  const histPuntos = hist.map((m, i) => {
    const diffMin = Math.round((new Date(m.timestamp) - new Date(hist[hist.length - 1].timestamp)) / 60000);
    return { label: diffMin === 0 ? 'ahora' : diffMin + 'm', real: m.nivel_cm, proyeccion: null, lstmPunto: null };
  });

  const proyPuntos = proy.puntos.slice(1).map(p => ({
    label: '+' + p.minutos + 'm',
    real: null,
    proyeccion: p.distancia_cm,
    lstmPunto: lstmEta !== null && Math.abs(p.minutos - lstmEta) < 6 ? p.distancia_cm : null
  }));

  const chartData = [...histPuntos, ...proyPuntos];

  // Interpretación combinada
  let interp = '';
  const prec = proy.cruces_umbrales?.precaucion;
  if (lstmEta !== null && prec) {
    const diff = Math.abs(lstmEta - prec);
    interp = `Proyección lineal: precaución en ${prec} min. LSTM (red neuronal): ${lstmEta} min. Diferencia: ${diff} min. ${diff < 30 ? 'Alta concordancia entre modelos.' : 'El LSTM considera factores estacionales adicionales.'}`;
  } else if (lstmEta !== null) {
    interp = `LSTM predice precaución en ${lstmEta} min. La proyección lineal no cruza el umbral en 120 min (río ${tasa > 0 ? 'bajando' : 'estable'}).`;
  } else {
    interp = `Tasa actual: ${tasa.toFixed(3)} cm/min. ${tasa < 0 ? 'Río subiendo — monitoreo activo.' : 'Río bajando o estable.'}`;
  }

  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '.07em' }}>
          Predicción LSTM + Proyección futura
        </span>
        <div style={{ display: 'flex', gap: 8 }}>
          <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: '#0f2d1a', color: '#22c55e', border: '1px solid #166534', fontWeight: 500 }}>LSTM activo</span>
          <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: '#1e3a5f', color: '#60a5fa', border: '1px solid #1d4ed8', fontWeight: 500 }}>120 min horizonte</span>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 14 }}>
        <div style={{ background: '#0f172a', borderRadius: 8, padding: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>LSTM — tiempo a precaución</div>
          <div style={{ fontFamily: 'monospace', fontSize: 32, fontWeight: 700, color: etaColor }}>{lstmEta !== null ? lstmEta : 'N/A'}</div>
          <div style={{ fontSize: 11, color: '#475569' }}>minutos (red neuronal)</div>
        </div>
        <div style={{ background: '#0f172a', borderRadius: 8, padding: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Proyección — precaución</div>
          <div style={{ fontFamily: 'monospace', fontSize: 32, fontWeight: 700, color: prec ? '#f59e0b' : '#22c55e' }}>{prec || '>120'}</div>
          <div style={{ fontSize: 11, color: '#475569' }}>minutos (lineal)</div>
        </div>
        <div style={{ background: '#0f172a', borderRadius: 8, padding: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', marginBottom: 6 }}>Tasa de cambio</div>
          <div style={{ fontFamily: 'monospace', fontSize: 32, fontWeight: 700, color: tasa < -0.5 ? '#ef4444' : tasa > 0.5 ? '#22c55e' : '#94a3b8' }}>
            {tasa > 0 ? '+' : ''}{tasa.toFixed(2)}
          </div>
          <div style={{ fontSize: 11, color: '#475569' }}>cm por minuto</div>
        </div>
      </div>

      {/* Leyenda */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 10, flexWrap: 'wrap' }}>
        {[
          { color: '#60a5fa', dash: false, label: 'Historial real' },
          { color: '#60a5fa', dash: true,  label: 'Proyección lineal' },
          { color: '#ef4444', dot: true,   label: 'Predicción LSTM' },
          { color: '#f59e0b', dash: true,  label: 'Precaución 200cm' },
          { color: '#ef4444', dash: true,  label: 'Alerta 100cm' },
        ].map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#64748b' }}>
            {l.dot
              ? <div style={{ width: 10, height: 10, borderRadius: '50%', background: l.color }} />
              : <div style={{ width: 20, height: 2, background: l.color, borderTop: l.dash ? `2px dashed ${l.color}` : 'none', opacity: l.dash ? 0.8 : 1 }} />
            }
            {l.label}
          </div>
        ))}
      </div>

      {/* Gráfica */}
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#475569' }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 9, fill: '#475569' }} axisLine={false} tickLine={false} tickFormatter={v => v + 'cm'} domain={[0, 350]} width={46} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={DIST_PRECAUCION} stroke="#f59e0b" strokeDasharray="4 3" strokeOpacity={0.7} />
          <ReferenceLine y={DIST_ALERTA}     stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.7} />
          <ReferenceLine y={DIST_EMERGENCIA} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.5} />
          <Line type="monotone" dataKey="real"       name="Historial"   stroke="#60a5fa" strokeWidth={2} dot={false} connectNulls={false} />
          <Line type="monotone" dataKey="proyeccion" name="Proyección"  stroke="#60a5fa" strokeWidth={2} strokeDasharray="6 4" dot={false} connectNulls={false} />
          <Line type="monotone" dataKey="lstmPunto"  name="LSTM"        stroke="#ef4444" strokeWidth={0} dot={{ r: 8, fill: '#ef4444', strokeWidth: 2, stroke: '#fff' }} connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>

      {/* Interpretación */}
      <div style={{ background: '#0f172a', borderLeft: '3px solid #3b82f6', borderRadius: '0 8px 8px 0', padding: '12px 14px', fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginTop: 12, marginBottom: 14 }}>
        {interp}
      </div>

      {/* Explicación diferencia */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div style={{ background: '#0f172a', borderRadius: 8, padding: 12, borderLeft: '3px solid #3b82f6' }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>Proyección (línea punteada)</div>
          <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>Extrapolación matemática lineal. Calcula cómo estará el río si continúa con la tasa de cambio actual.</div>
        </div>
        <div style={{ background: '#0f172a', borderRadius: 8, padding: 12, borderLeft: '3px solid #ef4444' }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>Predicción LSTM (punto rojo)</div>
          <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6 }}>Red neuronal entrenada con datos históricos del río. Considera estacionalidad, lluvias y patrones temporales.</div>
        </div>
      </div>
    </div>
  );
}

// ── Panel Comparación Histórica ───────────────────────────────
function PanelHistorico() {
  const [datos, setDatos] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDatos = useCallback(async () => {
    try {
      const res = await fetch(`${API}/comparacion-historica`);
      const data = await res.json();
      if (data.disponible) setDatos(data);
    } catch {}
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchDatos();
    const t = setInterval(fetchDatos, 60000);
    return () => clearInterval(t);
  }, [fetchDatos]);

  if (loading) return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
      <div style={{ fontSize: 11, color: '#475569', fontFamily: 'monospace' }}>Consultando datos históricos...</div>
    </div>
  );
  if (!datos) return null;

  const pct_peor   = Math.min(100, (datos.peor_caso_min / datos.mejor_caso_min) * 100);
  const pct_prom   = Math.min(100, (datos.min_promedio / datos.mejor_caso_min) * 100);

  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderTop: '3px solid #8b5cf6', borderRadius: 10, padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '.07em' }}>
          Comparación histórica — Dataset Río Mopán 2022-2024
        </span>
        <div style={{ display: 'flex', gap: 8 }}>
          <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: '#2d1f4e', color: '#a78bfa', border: '1px solid #7c3aed', fontWeight: 500 }}>
            {datos.n_episodios} episodios
          </span>
          <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: '#1e293b', color: '#64748b', border: '1px solid #334155', fontWeight: 500 }}>
            Ref: {datos.dist_referencia_cm} cm
          </span>
        </div>
      </div>

      {/* Métricas */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
        {[
          { label: 'Promedio histórico', val: datos.min_promedio, sub: 'minutos', color: '#8b5cf6' },
          { label: 'Mediana', val: datos.min_mediana, sub: 'minutos', color: '#60a5fa' },
          { label: 'Caso más rápido ⚠️', val: datos.mejor_caso_min, sub: 'minutos — más peligroso', color: '#ef4444' },
          { label: 'Caso más lento ✓', val: datos.peor_caso_min, sub: 'minutos — más seguro', color: '#22c55e' },
        ].map(m => (
          <div key={m.label} style={{ background: '#0f172a', borderRadius: 8, padding: 12, textAlign: 'center' }}>
            <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 6 }}>{m.label}</div>
            <div style={{ fontFamily: 'monospace', fontSize: 28, fontWeight: 700, color: m.color, lineHeight: 1 }}>{m.val}</div>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 3 }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* Barra visual de rango */}
      <div style={{ background: '#0f172a', borderRadius: 8, padding: 14, marginBottom: 14 }}>
        <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 10 }}>
          Rango histórico de tiempo a precaución
        </div>
        <div style={{ position: 'relative', height: 8, background: '#1e293b', borderRadius: 4, marginBottom: 8 }}>
          {/* Rango completo */}
          <div style={{ position: 'absolute', left: '0%', width: '100%', height: '100%', background: 'rgba(139,92,246,0.2)', borderRadius: 4 }} />
          {/* Promedio */}
          <div style={{
            position: 'absolute',
            left: `${Math.min(95, (datos.min_promedio / datos.mejor_caso_min) * 100)}%`,
            top: -4, width: 2, height: 16, background: '#8b5cf6', borderRadius: 1
          }} />
          {/* Peor caso */}
          <div style={{
            position: 'absolute',
            left: `${Math.min(95, (datos.peor_caso_min / datos.mejor_caso_min) * 100)}%`,
            top: -4, width: 2, height: 16, background: '#ef4444', borderRadius: 1
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#475569' }}>
          <span style={{ color: '#ef4444' }}>Mín: {datos.mejor_caso_min} min</span>
          <span style={{ color: '#8b5cf6' }}>Promedio: {datos.min_promedio} min</span>
          <span style={{ color: '#22c55e' }}>Máx: {datos.peor_caso_min} min</span>
        </div>
      </div>

      {/* Interpretación */}
      <div style={{ background: '#0f172a', borderLeft: '3px solid #8b5cf6', borderRadius: '0 8px 8px 0', padding: '12px 14px', fontSize: 12, color: '#94a3b8', lineHeight: 1.6, marginBottom: 10 }}>
        {datos.interpretacion}
      </div>

      <div style={{ fontSize: 10, color: '#334155', textAlign: 'right' }}>
        Mes más frecuente en episodios similares: {datos.mes_mas_comun}
      </div>
    </div>
  );
}

// ── Simulador ─────────────────────────────────────────────────
function Simulador({ onEnvio }) {
  const [dist, setDist] = useState(250);
  const [voltaje, setVoltaje] = useState(12.65);
  const [dispositivo, setDispositivo] = useState('SAT-MOPAN-01');
  const [loading, setLoading] = useState(false);
  const [log, setLog] = useState('Listo para simular.');
  const [logColor, setLogColor] = useState('#64748b');

  const est = getEstado(dist);
  const pct = Math.min(100, (dist / DIST_NORMAL) * 100);

  const enviar = async () => {
    setLoading(true);
    setLog('Enviando...');
    setLogColor('#94a3b8');
    try {
      const res = await fetch(`${API}/medicion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nivel_cm: dist, voltaje_bateria: voltaje, codigo_estado: 0, dispositivo_id: dispositivo })
      });
      const data = await res.json();
      if (res.status === 201) {
        setLogColor('#22c55e');
        setLog(`OK (201) — ID: ${data.id} | Distancia: ${dist}cm | ${new Date().toLocaleTimeString('es-GT')}`);
        setTimeout(onEnvio, 1000);
      } else {
        setLogColor('#f97316');
        setLog('Error: ' + JSON.stringify(data));
      }
    } catch {
      setLogColor('#ef4444');
      setLog('Error: No se pudo conectar con el servidor.');
    }
    setLoading(false);
  };

  const presets = [
    { label: 'Normal (280cm)',      v: 280, s: {} },
    { label: 'Normal bajo (220cm)', v: 220, s: {} },
    { label: 'Precaución (180cm)',  v: 180, s: { borderColor: '#f59e0b', color: '#f59e0b' } },
    { label: 'Alerta (90cm)',       v: 90,  s: { borderColor: '#f97316', color: '#f97316' } },
    { label: 'Emergencia (40cm)',   v: 40,  s: { borderColor: '#ef4444', color: '#ef4444' } },
    { label: 'Error sensor (999)',  v: 999, s: { borderColor: '#64748b', color: '#64748b' } },
  ];

  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderTop: '3px solid #7c3aed', borderRadius: 10, padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#a78bfa', textTransform: 'uppercase', letterSpacing: '.07em' }}>Simulador de mediciones</div>
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Simula distancia del sensor al agua (menor = más peligro)</div>
        </div>
        <span style={{ fontSize: 10, padding: '3px 8px', borderRadius: 20, background: '#2d1f4e', color: '#a78bfa', border: '1px solid #7c3aed', fontWeight: 500 }}>Solo pruebas</span>
      </div>

      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 10, color: '#7c3aed', textTransform: 'uppercase', letterSpacing: '.08em', fontWeight: 500, marginBottom: 8 }}>Distancia sensor al agua (cm) — menor = mayor peligro</div>
        <input type="range" min={20} max={350} step={1} value={dist} onChange={e => setDist(parseFloat(e.target.value))} style={{ width: '100%', accentColor: '#7c3aed', marginBottom: 6 }} />
        <div style={{ fontFamily: 'monospace', fontSize: 28, fontWeight: 700, color: est.c, textAlign: 'center', margin: '6px 0 4px' }}>{dist} cm</div>
        <div style={{ height: 8, background: '#0f172a', borderRadius: 4, overflow: 'hidden', marginBottom: 4 }}>
          <div style={{ height: '100%', width: pct + '%', background: est.c, borderRadius: 4, transition: 'all .3s' }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#475569' }}>
          <span>Emergencia &lt;50cm</span><span>Alerta &lt;100cm</span><span>Precaución &lt;200cm</span><span>Normal &gt;200cm</span>
        </div>
      </div>

      <div style={{ fontSize: 10, color: '#7c3aed', textTransform: 'uppercase', letterSpacing: '.08em', fontWeight: 500, marginBottom: 8 }}>Escenarios rápidos</div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
        {presets.map(p => (
          <button key={p.v} onClick={() => setDist(p.v)} style={{ background: '#0f172a', border: '1px solid #334155', color: '#94a3b8', borderRadius: 6, padding: '5px 12px', fontSize: 11, cursor: 'pointer', fontFamily: 'inherit', ...p.s }}>{p.label}</button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
        {[
          { label: 'Voltaje (V)', type: 'number', val: voltaje, set: e => setVoltaje(parseFloat(e.target.value)), min: 10, max: 14, step: 0.01 },
          { label: 'Dispositivo ID', type: 'text', val: dispositivo, set: e => setDispositivo(e.target.value) },
        ].map(f => (
          <div key={f.label}>
            <label style={{ fontSize: 10, color: '#7c3aed', textTransform: 'uppercase', letterSpacing: '.08em', fontWeight: 500, display: 'block', marginBottom: 6 }}>{f.label}</label>
            <input type={f.type} value={f.val} onChange={f.set} min={f.min} max={f.max} step={f.step}
              style={{ width: '100%', background: '#0f172a', border: '1px solid #334155', borderRadius: 6, padding: '8px 10px', color: '#f1f5f9', fontSize: 13, fontFamily: 'monospace', outline: 'none' }} />
          </div>
        ))}
      </div>

      <button onClick={enviar} disabled={loading} style={{ width: '100%', background: loading ? '#374151' : '#7c3aed', border: 'none', color: loading ? '#6b7280' : '#fff', borderRadius: 8, padding: 10, fontSize: 13, fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer', fontFamily: 'inherit', marginBottom: 10 }}>
        {loading ? 'Enviando...' : 'Enviar medición al servidor'}
      </button>
      <div style={{ background: '#0f172a', borderRadius: 6, padding: 10, fontFamily: 'monospace', fontSize: 11, color: logColor }}>{log}</div>
    </div>
  );
}

// ── App principal ─────────────────────────────────────────────
export default function App() {
  const [actual, setActual] = useState(null);
  const [historial, setHistorial] = useState([]);
  const [loading, setLoading] = useState(true);
  const [liveStatus, setLiveStatus] = useState('EN VIVO');
  const [hora, setHora] = useState('--:--:--');

  useEffect(() => {
    const t = setInterval(() => setHora(new Date().toLocaleTimeString('es-GT', { hour: '2-digit', minute: '2-digit', second: '2-digit' })), 1000);
    return () => clearInterval(t);
  }, []);

  const fetchData = useCallback(async () => {
    try {
      setLiveStatus('ACTUALIZANDO');
      const [rA, rH] = await Promise.all([fetch(`${API}/nivel-actual`), fetch(`${API}/historial?limite=40`)]);
      if (!rA.ok) throw new Error();
      const dataActual = await rA.json();
      const dataHist = (await rH.json()).reverse();
      setActual(dataActual);
      setHistorial(dataHist);
      setLoading(false);
      setLiveStatus('EN VIVO');
    } catch {
      setLiveStatus('SIN CONEXIÓN');
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); const t = setInterval(fetchData, 30000); return () => clearInterval(t); }, [fetchData]);

  const distCm  = actual?.nivel_cm ?? 999;
  const voltaje = actual?.voltaje_bateria ?? 0;
  const est     = getEstado(distCm);
  const pctBarra = distCm < 900 ? Math.min(100, (distCm / DIST_NORMAL) * 100) : 0;

  const chartData = historial.filter(m => m.nivel_cm < 900).map(m => ({
    hora: new Date(m.timestamp).toLocaleTimeString('es-GT', { hour: '2-digit', minute: '2-digit' }),
    dist: parseFloat(m.nivel_cm.toFixed(1))
  }));

  const tablaData = historial.filter(m => m.nivel_cm < 900).slice(-8).reverse();

  return (
    <div style={{ minHeight: '100vh', background: '#111827', color: '#f1f5f9', fontFamily: "'Segoe UI', sans-serif" }}>
      {/* Header */}
      <div style={{ background: '#0f172a', borderBottom: '1px solid #1e293b', padding: '14px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 100 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 32, height: 32, background: '#1d4ed8', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>SAT Mopán — Panel de Monitoreo</div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 1 }}>Río Mopán · Puente El Camalote · Melchor de Mencos, Petén</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#0f2d1a', border: '1px solid #166534', borderRadius: 20, padding: '4px 10px' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: liveStatus === 'SIN CONEXIÓN' ? '#ef4444' : '#22c55e' }} />
            <span style={{ fontSize: 11, color: liveStatus === 'SIN CONEXIÓN' ? '#ef4444' : '#22c55e', fontWeight: 500 }}>{liveStatus}</span>
          </div>
          <div style={{ fontSize: 11, color: '#475569', textAlign: 'right' }}>
            <div>{hora}</div><div style={{ marginTop: 1 }}>SAT-MOPAN-01</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14 }}>
        {loading ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '4rem', fontFamily: 'monospace', fontSize: 13 }}>Conectando con el servidor...</div>
        ) : (
          <>
            {/* Banner */}
            <div style={{ background: est.bg, border: `1px solid ${est.bc}`, borderRadius: 10, padding: '12px 18px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: est.c }}>Estado del río: {est.l}</span>
              <span style={{ fontSize: 20, fontWeight: 700, color: est.c, fontFamily: 'monospace' }}>
                {distCm < 900 ? distCm.toFixed(1) + ' cm' : 'Sin lectura'}
              </span>
            </div>

            {/* KPIs */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 10 }}>
              <KpiCard accent="#3b82f6" label="Distancia al agua" value={distCm < 900 ? distCm.toFixed(1) + ' cm' : 'Error'} color={est.c} sub="menor distancia = mayor riesgo" />
              <KpiCard accent="#f59e0b" label="Batería" value={voltaje.toFixed(2)} color="#f59e0b" sub={voltaje >= 12 ? 'carga óptima' : 'carga baja'} />
              <KpiCard accent="#8b5cf6" label="Última lectura" value={actual ? new Date(actual.timestamp).toLocaleString('es-GT', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '--/--/----'} color="#8b5cf6" valueSub sub="actualiza automáticamente" />
              <KpiCard accent="#22c55e" label="Mediciones" value={historial.length} color="#22c55e" sub="en este historial" />
            </div>

            {/* Gráfica + Gauge */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 14 }}>
              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '.07em' }}>Historial — distancia sensor al agua</span>
                  <div style={{ display: 'flex', gap: 14 }}>
                    {[['#3b82f6','distancia'],['#f59e0b','precaución 200cm'],['#ef4444','alerta 100cm']].map(([c,l]) => (
                      <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: '#64748b' }}>
                        <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />{l}
                      </div>
                    ))}
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="distGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={est.c} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={est.c} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="hora" tick={{ fontSize: 10, fill: '#475569' }} axisLine={{ stroke: '#1e293b' }} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 10, fill: '#475569' }} axisLine={false} tickLine={false} tickFormatter={v => v + 'cm'} domain={[0, 350]} width={50} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={DIST_PRECAUCION} stroke="#f59e0b" strokeDasharray="4 3" strokeOpacity={0.7} label={{ value: '200cm', fill: '#f59e0b', fontSize: 9 }} />
                    <ReferenceLine y={DIST_ALERTA}     stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.7} label={{ value: '100cm', fill: '#ef4444', fontSize: 9 }} />
                    <ReferenceLine y={DIST_EMERGENCIA} stroke="#ef4444" strokeDasharray="4 3" strokeOpacity={0.9} label={{ value: '50cm',  fill: '#ef4444', fontSize: 9 }} />
                    <Area type="monotone" dataKey="dist" stroke={est.c} strokeWidth={2} fill="url(#distGrad)" dot={false} activeDot={{ r: 4, fill: est.c, strokeWidth: 0 }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 14 }}>Distancia al agua</div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                  <div style={{ fontFamily: 'monospace', fontSize: 32, fontWeight: 700, color: est.c }}>{distCm < 900 ? distCm.toFixed(1) : '--'} cm</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>distancia sensor → agua</div>
                  <div style={{ width: '100%' }}>
                    {[
                      { label: 'Normal',     color: '#22c55e', val: distCm > DIST_PRECAUCION ? 100 : 0, ref: '>200cm' },
                      { label: 'Precaución', color: '#f59e0b', val: distCm <= DIST_PRECAUCION && distCm > DIST_ALERTA ? 100 : 0, ref: '<200cm' },
                      { label: 'Alerta',     color: '#f97316', val: distCm <= DIST_ALERTA && distCm > DIST_EMERGENCIA ? 100 : 0, ref: '<100cm' },
                      { label: 'Emergencia', color: '#ef4444', val: distCm <= DIST_EMERGENCIA ? 100 : 0, ref: '<50cm' },
                    ].map(g => (
                      <div key={g.label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11, marginBottom: 8 }}>
                        <span style={{ color: g.color, width: 72 }}>{g.label}</span>
                        <div style={{ flex: 1, height: 6, background: '#0f172a', borderRadius: 3, margin: '0 10px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: g.val + '%', background: g.color, borderRadius: 3, transition: 'width .6s ease' }} />
                        </div>
                        <span style={{ color: '#475569', width: 40, textAlign: 'right', fontSize: 10 }}>{g.ref}</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ width: '100%', height: 8, background: '#0f172a', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: pctBarra + '%', background: est.c, borderRadius: 4, transition: 'width .8s ease' }} />
                  </div>
                  <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#475569' }}>
                    <span>0cm</span><span>150cm</span><span>300cm</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Panel ML Random Forest */}
            <PanelML />

            {/* Panel Predicción + Proyección LSTM */}
            <PanelPrediccionProyeccion />

            {/* Panel Comparación Histórica */}
            <PanelHistorico />

            {/* Simulador */}
            <Simulador onEnvio={fetchData} />

            {/* Tabla */}
            <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '.07em' }}>Últimas mediciones registradas</span>
                <span style={{ fontSize: 10, color: '#475569' }}>{historial.length} registros</span>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>{['Hora', 'Distancia', 'Voltaje', 'Estado'].map(h => (
                    <th key={h} style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: '.07em', padding: '6px 8px', borderBottom: '1px solid #1e293b', textAlign: 'left', fontWeight: 500 }}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {tablaData.length === 0 ? (
                    <tr><td colSpan={4} style={{ textAlign: 'center', color: '#475569', padding: 20, fontSize: 12 }}>Sin datos</td></tr>
                  ) : tablaData.map(m => {
                    const e = getEstado(m.nivel_cm);
                    return (
                      <tr key={m.id}>
                        <td style={{ fontSize: 12, padding: '7px 8px', borderBottom: '1px solid #1e293b', color: '#f1f5f9', fontFamily: 'monospace' }}>
                          {new Date(m.timestamp).toLocaleString('es-GT', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td style={{ fontSize: 12, padding: '7px 8px', borderBottom: '1px solid #1e293b', color: '#60a5fa', fontWeight: 600, fontFamily: 'monospace' }}>{m.nivel_cm.toFixed(1)} cm</td>
                        <td style={{ fontSize: 12, padding: '7px 8px', borderBottom: '1px solid #1e293b', color: '#f59e0b', fontFamily: 'monospace' }}>{m.voltaje_bateria.toFixed(2)} V</td>
                        <td style={{ fontSize: 12, padding: '7px 8px', borderBottom: '1px solid #1e293b' }}>
                          <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 20, background: e.bg, color: e.c, border: `1px solid ${e.bc}` }}>{e.l}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Footer */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', borderTop: '1px solid #1e293b' }}>
              <span style={{ fontSize: 10, color: '#475569' }}>Universidad Mariano Gálvez · Héctor Daniel Pérez 5190-15-3835 · SAT Río Mopán</span>
              <button onClick={fetchData} style={{ background: '#1d4ed8', border: 'none', color: '#fff', borderRadius: 6, padding: '6px 14px', fontSize: 11, cursor: 'pointer', fontWeight: 500, fontFamily: 'inherit' }}>Actualizar ↻</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
