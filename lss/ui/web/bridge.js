// LSS bridge — connects the frozen Claude Design component (40a) to the local backend.
//
// This file is appended inside the design's component script, right after
// `class Component`, and replaces that binding with a subclass. It never changes
// how anything looks or moves. It only swaps:
//   - demo data (BASE / GEN / FORMATS / demo fault metrics / synthetic source wave / HAVE0 / f.bgm)
//     for backend facts
//   - demo timing (setInterval step, demoSpeed) for backend progress and settle events
//   - demo actions (addMany pool, instant removal, local outcome()) for backend commands
//   - the SOUND board's words go to START with the wanted sets (the backend freezes them per SOURCE)
//   - the prototype's stand-ins for OS-owned surfaces (desktop, tray, notification card,
//     "LSS を終了しました" screen) for the real tray, notification and process exit
// Everything else — phase transitions, wipes, stamps, the separation wave, the landing pass,
// selection, marks, the OUTPUT selector (common setting -> per-SOURCE setting), the quit
// confirmation card, menus, tooltips, keyboard — is the design's own code. The backend freezes
// the resolved wanted set per SOURCE at START, owns per-result outcomes (tst), existing results
// (have) and the session's frozen common setting.
Component = class LSSComponent extends Component {
  constructor(props) {
    super(props);
    this._hot = true;
    this._songs = {};
    this._lss = {
      settleSeq: 0, arrivalSeq: 0, faultSeq: 0,
      waves: {}, waveAsked: {}, toast: [],
      swapUntil: 0, deferred: null, seen: null,
      afterSkip: false, choosing: false, completeSent: false, starting: false
    };
    // demo fault figures are not facts. only measured numbers may appear.
    const faults = {};
    Object.keys(this.FAULTS).forEach(k => { faults[k] = Object.assign({}, this.FAULTS[k], { metrics: [] }); });
    this.FAULTS = faults;
    this._rowErrBase = this.ROW_ERR;
    const init = window.__LSS_INITIAL__;
    Object.assign(this.state, { queue: [], hist: [], st: {}, tst: {}, have: {}, frozen: {}, frozenDef: null });
    if (init) {
      this._lss.seen = init;
      this._lss.settleSeq = init.settle.seq;
      this._lss.arrivalSeq = init.arrival.seq;
      this._lss.faultSeq = init.fault ? init.fault.seq : 0;
      // the common setting and per-SOURCE settings the backend kept from the last START
      const bdef = this.norm(init.bdef || ['bgm']);
      const ovr = {};
      Object.keys(init.wants || {}).forEach(id => {
        const w = this.norm(init.wants[id]);
        if (w.length && w.join() !== bdef.join()) ovr[id] = w;
      });
      Object.assign(this.state, { bdef: bdef.length ? bdef : ['bgm'], ovr, sounds: typeof init.soundsText === 'string' ? init.soundsText : '' });
      Object.assign(this.state, this.lssFacts(init, this.state));
    }
  }

  // ---- backend facts -> the design's own state vocabulary ----
  lssFacts(snap, s) {
    this._songs = snap.songs || {};
    this.ROW_ERR = Object.assign({}, this._rowErrBase, snap.errs || {});
    const queue = snap.queue || [];
    const st = Object.assign({}, snap.st || {});
    const fault = snap.fault || null;
    let ci = s.ci;
    if (fault && fault.item && queue.indexOf(fault.item) >= 0) {
      // the halted source stays the current row until RETRY / SKIP decides it
      ci = queue.indexOf(fault.item);
      st[fault.item] = 'queued';
    } else if (snap.running && queue.indexOf(snap.running) >= 0) {
      ci = queue.indexOf(snap.running);
    } else {
      const next = queue.findIndex(id => st[id] === 'queued');
      if (next >= 0) ci = next;
      else if (ci >= queue.length) ci = Math.max(0, queue.length - 1);
    }
    const PH = { PREP: 0, SEPARATE: 1, SAVE: 2 };
    const stage = snap.stage;
    // a SOURCE's wanted set is frozen only while a session holds it (START .. BACK TO READY / STOP)
    const frozenDef = snap.frozenDef ? this.norm(snap.frozenDef) : null;
    const frozen = {};
    if (frozenDef) Object.keys(snap.wants || {}).forEach(id => { if (queue.indexOf(id) >= 0) frozen[id] = this.norm(snap.wants[id]); });
    const patch = {
      queue, hist: snap.hist || [], st, ci,
      tst: snap.tst || {}, have: snap.have || {}, frozen, frozenDef,
      ph: stage ? (PH[stage.k] || 0) : 0,
      phP: stage && typeof stage.p === 'number' ? stage.p : 0,
      mcached: !snap.firstRun, dlDone: false,
      elapsed: snap.elapsed || 0, curEl: snap.curElapsed || 0,
      forecast: snap.forecast || null,
      fault: fault ? fault.key : null
    };
    // the SOURCE that was being worked on reached its end: it lands once (gold, red on failure)
    if (s.phase === 'processing') {
      const was = (s.queue || [])[s.ci];
      const before = (s.st || {})[was];
      if (was && (!before || before === 'queued') && st[was] && st[was] !== 'queued') {
        patch.landId = was;
        patch.landKey = (s.landKey || 0) + 1;
      }
    }
    if (s.marks && s.marks.length) {
      const m = s.marks.filter(id => queue.indexOf(id) >= 0);
      if (m.length !== s.marks.length) patch.marks = m;
    }
    if (fault && fault.seq !== this._lss.faultSeq) patch.faultKey = (s.faultKey || 0) + 1;
    this._lss.faultSeq = fault ? fault.seq : this._lss.faultSeq;
    if (s.selId && queue.indexOf(s.selId) < 0 && patch.hist.indexOf(s.selId) < 0) patch.selId = null;
    return patch;
  }

  lssApply(snap) {
    const L = this._lss;
    L.seen = snap;
    // a state swap is hidden under the wipe: facts that arrive meanwhile land with the swap.
    // while START is being answered the phase is not decided yet, so facts wait for it too.
    if (L.starting || performance.now() < L.swapUntil) { L.deferred = snap; return; }
    L.deferred = null;
    const s = this.state;
    const patch = this.lssFacts(snap, s);
    let after = null;
    if (snap.settle && snap.settle.seq !== L.settleSeq) {
      L.settleSeq = snap.settle.seq;
      if (s.phase === 'processing') {
        if (snap.settle.kind === 'drained') {
          patch.allDone = true;
          const wait = L.afterSkip ? 700 : 900;
          after = () => {
            if (this._fin) return;
            this._fin = true;
            clearTimeout(this._t.d);
            this._t.d = setTimeout(() => this.wipeTo('complete'), wait);
          };
        } else if (snap.settle.kind === 'stopped') {
          Object.assign(patch, { phase: 'ready', stopped: true, stopKind: 'stop', selId: null, stopReq: false, landId: null });
        }
      }
      L.afterSkip = false;
    }
    if (snap.arrival && snap.arrival.seq !== L.arrivalSeq) {
      L.arrivalSeq = snap.arrival.seq;
      const add = (snap.arrival.ids || []).filter(id => patch.queue.indexOf(id) >= 0);
      if (add.length) {
        patch.newIds = add;
        patch.ctx = null;
        const prev = after;
        after = () => {
          if (prev) prev();
          clearTimeout(this._t.n); this._t.n = setTimeout(() => this.setState({ newIds: null }), 900);
          clearTimeout(this._t.ns); this._t.ns = setTimeout(() => this.scrollTo(add[add.length - 1]), 40);
        };
      }
    }
    this.setState(patch, after);
  }

  lssCall(name, ...args) {
    const api = window.pywebview && window.pywebview.api;
    if (!api || typeof api[name] !== 'function') return Promise.resolve(null);
    return api[name](...args).catch(err => { console.error('[lss] ' + name, err); return null; });
  }

  componentDidMount() {
    super.componentDidMount();
    window.__lssComponent = this;
    window.__lssPush = (snap) => this.lssApply(snap);
    const latest = window.__LSS_INITIAL__;
    if (latest && latest !== this._lss.seen) this.lssApply(latest);
    window.__lssSystemScale = (scale) => this.lssCursors(scale);
    this.lssCursors(1);
    this.lssReportUi();
    // the bridge may mount before pywebview's api exists: report the screen state again once it does
    const ready = () => { this._lss.uiKey = null; this.lssReportUi(); this.lssCall('ready'); this.lssAskWaves(); };
    if (window.pywebview && window.pywebview.api) ready();
    else window.addEventListener('pywebviewready', ready, { once: true });
  }

  componentDidUpdate(prevProps) {
    super.componentDidUpdate(prevProps);
    const s = this.state, L = this._lss;
    this.lssReportUi();
    if (s.phase === 'complete') {
      // the OS notification (only while LSS is in the tray) says what the COMPLETE panel says
      if (!L.completeSent) { L.completeSent = true; this.lssCall('sessionComplete', s.post, L.toast.slice()); }
    } else {
      L.completeSent = false;
    }
    this.lssAskWaves();
  }

  // measured peaks for the open row and for the SOURCE being worked on (120 bars each)
  lssAskWaves() {
    const s = this.state, L = this._lss;
    const ids = [s.selId, s.phase === 'processing' ? s.queue[s.ci] : null];
    ids.forEach(id => {
      if (!id || !this._songs[id] || L.waves[id] || L.waveAsked[id]) return;
      L.waveAsked[id] = true;
      this.lssCall('wave', id, 120).then(values => {
        if (Array.isArray(values) && values.length) { L.waves[id] = values; this.forceUpdate(); }
      });
    });
  }

  // ---- System Assets (Claude Design): tray state and cursor family ----
  lssReportUi() {
    const s = this.state;
    const ui = { phase: s.phase, fault: !!s.fault, stopReq: !!s.stopReq, allDone: !!s.allDone };
    const key = JSON.stringify(ui);
    if (key !== this._lss.uiKey) { this._lss.uiKey = key; this.lssCall('uiState', ui); }
    const frame = this._frame;
    if (!frame) return;
    const working = s.phase === 'processing' && !s.fault;
    frame.classList.toggle('lss-working', working);
    if (working && !this._lss.workTimer) {
      // the badge kerf steps 4 -> 5 px once per second; the arrow never moves
      this._lss.workTimer = setInterval(() => frame.classList.toggle('lss-f1'), 1000);
    } else if (!working && this._lss.workTimer) {
      clearInterval(this._lss.workTimer); this._lss.workTimer = null; frame.classList.remove('lss-f1');
    }
  }

  lssCursors(systemScale) {
    // one image pixel = one physical pixel: 32 / 40 / 48 px canvases for 100 / 125 / 150 %
    const sizes = [[1, 32], [1.25, 40], [1.5, 48]];
    const size = sizes.reduce((best, cur) => Math.abs(cur[0] - systemScale) < Math.abs(best[0] - systemScale) ? cur : best)[1];
    const dpr = window.devicePixelRatio || 1;
    const cur = (name, ink, f, fallback) => {
      const url = `system/cursor-${name}${ink ? '-ink' : ''}-${size}-f${f}.png`;
      return `cursor:url(${url}) 0 0,${fallback}!important;cursor:image-set(url(${url}) ${dpr}x) 0 0,${fallback}!important`;
    };
    const F = '[data-lss-frame]';
    const rules = [];
    for (const ink of [false, true]) {
      const on = ink ? '.lss-ink' : '';
      rules.push(`${F}${on},${F}${on} [style*="cursor: default"]{${cur('default', ink, 0, 'default')}}`);
      rules.push(`${F}${on} [style*="cursor: pointer"]{${cur('interactive', ink, 0, 'pointer')}}`);
      rules.push(`${F}.lss-working${on},${F}.lss-working${on} [style*="cursor: default"]{${cur('working', ink, 0, 'default')}}`);
      rules.push(`${F}.lss-working.lss-f1${on},${F}.lss-working.lss-f1${on} [style*="cursor: default"]{${cur('working', ink, 1, 'default')}}`);
    }
    let el = document.getElementById('lss-cursors');
    if (!el) { el = document.createElement('style'); el.id = 'lss-cursors'; document.head.appendChild(el); }
    el.textContent = rules.join('\n');
    if (!this._lss.inkWatch) {
      // over saturated LSS surfaces (red, cyan, gold) the backing plane switches to ink
      this._lss.inkWatch = true;
      document.addEventListener('mouseover', (e) => {
        const frame = this._frame;
        if (!frame) return;
        let node = e.target, ink = false;
        while (node && node !== document.body) {
          const m = getComputedStyle(node).backgroundColor.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?/);
          if (m && (m[4] === undefined || Number(m[4]) >= 0.5)) {
            const c = [Number(m[1]), Number(m[2]), Number(m[3])];
            ink = Math.max(...c) - Math.min(...c) > 90 && Math.max(...c) > 120;
            break;
          }
          node = node.parentElement;
        }
        frame.classList.toggle('lss-ink', ink);
      });
    }
  }

  songs() { return this._songs; }
  plan() { return [{ k: 'PREP' }, { k: 'SEPARATE' }, { k: 'SAVE' }]; }

  // The source lane is the SOURCE's measured overview (local peaks). The result lane below it
  // (`id + 'x'`) is the design's fixed display profile: it never previews the result
  // (visual spec 7.5). Without measured peaks only the SOURCE being worked on draws, as an
  // even low skeleton with the same motion (handoff 10.2); other rows draw nothing.
  wave(id, len) {
    const n = len || 44;
    if (this._songs[id]) {
      const measured = this._lss.waves[id];
      if (measured && measured.length) return this.lssFit(measured, n);
      const s = this.state;
      return s.phase === 'processing' && s.queue[s.ci] === id ? Array.from({ length: n }, () => 22) : [];
    }
    if (id.slice(-1) === 'x' && this._songs[id.slice(0, -1)]) return super.wave(id, n);
    return [];
  }
  lssFit(values, n) {
    if (values.length === n) return values;
    return Array.from({ length: n }, (_, i) => {
      const a = Math.floor(i * values.length / n), b = Math.max(a + 1, Math.floor((i + 1) * values.length / n));
      return Math.max(...values.slice(a, b));
    });
  }

  wipeTo(phase, extra, dir) {
    super.wipeTo(phase, extra, dir);
    const L = this._lss;
    L.swapUntil = performance.now() + 250;
    clearTimeout(this._t.lssSwap);
    this._t.lssSwap = setTimeout(() => {
      L.swapUntil = 0;
      if (L.deferred) this.lssApply(L.deferred);
    }, 251);
  }

  // START freezes each SOURCE's resolved wanted set in the backend. SOURCES whose every wanted
  // result already exists resolve there without processing; with no work left the session
  // goes straight to COMPLETE and never shows PROCESSING.
  start() {
    const s = this.state, L = this._lss;
    if (L.starting) return;
    const pending = s.queue.filter(id => !s.st[id] || s.st[id] === 'queued');
    const shake = () => this.setState(t => ({ shakeKey: t.shakeKey + 1 }));
    if (!pending.length) { shake(); return; }
    const wanted = {};
    pending.forEach(id => { wanted[id] = this.want(id, s).slice(); });
    // STEM needs at least one chosen stem: ask for it instead of starting
    const needsWords = pending.some(id => wanted[id].indexOf('snd') >= 0) && !(s.sounds || '').trim();
    if (needsWords) { this.setState(t => ({ soundNeed: true, shakeKey: t.shakeKey + 1 })); return; }
    L.starting = true;
    this.lssCall('start', wanted, s.bdef.slice(), s.sounds || '').then(res => {
      L.starting = false;
      const later = L.deferred;
      if (res && res.reason === 'sounds_required') this.setState({ soundNeed: true });
      if (!res || !res.started) { shake(); if (later) this.lssApply(later); return; }
      this._fin = false;
      L.afterSkip = false;
      const common = { marks: [], markAnchor: null, selId: null, fault: null, landId: null, stopReq: false, stopped: false, allDone: false, ph: 0, phP: 0, curEl: 0 };
      if (!res.work) { this.wipeTo('complete', common); return; }
      const first = pending.find(id => { const t = later && later.st ? later.st[id] : null; return !t || t === 'queued'; }) || pending[0];
      this.wipeTo('processing', Object.assign(common, { ci: s.queue.indexOf(first) }));
    });
  }
  requestStop() { super.requestStop(); this.lssCall('stop'); }
  backToReady() {
    const s = this.state;
    const ovr = {};
    s.queue.filter(id => s.st[id] === 'failed').forEach(id => { if (s.ovr[id]) ovr[id] = s.ovr[id]; });
    this.wipeTo('ready', { tst: {}, frozen: {}, frozenDef: null, ovr, marks: [], allDone: false, stopped: false, selId: null, fault: null, landId: null, quitAsk: false }, 'back');
    this.lssCall('backToReady').then(snap => { if (snap) this.lssApply(snap); });
  }
  clearFault(resume) {
    this.setState({ fault: null });
    if (resume) this.lssCall('haltRetry');
  }
  skipCurrent() {
    this._lss.afterSkip = true;
    this.setState({ fault: null });
    this.lssCall('haltSkip');
  }
  addFile() { this.lssChoose('files'); }
  addFolder() { this.lssChoose('folder'); }
  lssChoose(kind) {
    const L = this._lss;
    if (L.choosing) return;
    L.choosing = true;
    if (this.state.ctx) this.setState({ ctx: null });
    this.lssCall('choose', kind).then(() => { L.choosing = false; });
  }
  removeFile(id) {
    this.setState(s => {
      const ovr = Object.assign({}, s.ovr); delete ovr[id];
      return { ovr, marks: s.marks.filter(x => x !== id), selId: s.selId === id ? null : s.selId, ctx: null };
    });
    this.lssCall('remove', id);
  }
  retry(id) {
    this.setState({ selId: null, ctx: null });
    this.lssCall('retryRow', id);
  }

  // ---- window: the OS owns the tray, the notification and the process ----
  // minimize = hide into the real tray; work continues. The tray icon restores full screen.
  toTray() {
    this.setState({ quitAsk: false, tip: null, ctx: null });
    this.lssCall('minimize');
  }
  restore() {}
  // close = quit the app. requestQuit (the design's own) asks once only while a SOURCE is in
  // flight; the backend then stops it before publication, so nothing partial is saved.
  quit() {
    this.setState({ quitAsk: false, tip: null, ctx: null });
    this.lssCall('quit');
  }
  lssOpen(path) { if (path) this.lssCall('openFolder', path); }

  renderVals() {
    const v = super.renderVals();
    const s = this.state;
    // the app window is the frame: keys are live wherever the pointer is
    v.onEnterFrame = () => { this._hot = true; };
    v.onLeaveFrame = () => this.exitTip();
    const ctxId = s.ctx && s.ctx.id;
    v.ctxItems.forEach(it => {
      if (it.key === 'copy') it.onClick = () => { this.lssCall('copyText', this.song(ctxId).dir); this.closeCtx(); };
      if (it.key === 'open') it.onClick = () => { this.lssOpen(this.song(ctxId).dir); this.closeCtx(); };
    });
    // with no destination yet, the design falls back to a demo folder. the honest statement is the rule.
    const dirsOf = (ids) => ids.map(id => this.song(id).dir).filter(Boolean);
    const pending = s.queue.filter(id => !s.st[id] || s.st[id] === 'queued');
    if (!dirsOf(pending.length ? pending : s.queue).length) { v.outPathRoot = 'EACH SOURCE FOLDER\\'; v.outPathLeaf = 'LSS'; }
    // the COMPLETE destination names only folders that received a saved file. held BGM
    // candidates stay internal (not published), so they never claim an LSS folder.
    if (s.phase === 'complete') {
      const saved = [];
      s.queue.forEach(id => { const f = this.song(id); if (f.out && f.dir && saved.indexOf(f.dir) < 0) saved.push(f.dir); });
      if (saved.length === 1) { const sp = this.splitPath(saved[0] + '\\LSS'); v.resPathRoot = sp.root; v.resPathLeaf = sp.leaf; }
      else { v.resPathRoot = 'EACH SOURCE FOLDER\\'; v.resPathLeaf = 'LSS'; }
      v.resMulti = saved.length > 1;
      v.resDestCount = saved.length + ' FOLDERS';
      v.resDestNote = saved.length + ' か所に保存しました。各音源の隣の LSS フォルダです。';
      // only one real folder can be opened; a pattern for several folders is not a place
      const one = saved.length === 1 ? saved[0] + '\\LSS' : null;
      v.resPathTip = one ? 'open' : '';
      v.resPathBoxStyle = one ? Object.assign({}, v.pathBoxStyle, { cursor: 'pointer', transition: 'border-color .15s ease, transform .12s ease' }) : v.pathBoxStyle;
      v.onOpenRes = (e) => { if (e) e.stopPropagation(); if (one) this.lssOpen(one); };
    }
    // what the COMPLETE panel counts, for the notification that may say it while LSS is in the tray
    this._lss.toast = (v.toastTags || []).map(t => t.t);
    // a row's path opens the folder it names: the source's folder, or its LSS folder once something was written there
    v.rows.forEach(r => {
      if (!r.folder) return;
      r.onOpen = (e) => { if (e) e.stopPropagation(); this.lssOpen(r.folder); };
    });
    // published BGM: show the real file names (they carry the start time), and how many
    // stayed held (not saved) because they did not reach the standard.
    v.rows.forEach(r => {
      const o = (s.tst[r.key] || {}).bgm;
      if (!o || (o.k !== 'done' && o.k !== 'reused') || !(o.n > 0) || !Array.isArray(r.outputs)) return;
      const files = Array.isArray(o.files) ? o.files : [];
      let j = 0;
      r.outputs.forEach(x => {
        if (typeof x.key !== 'string' || x.key.indexOf('bgm') !== 0 || x.key === 'bgmmore') return;
        if (files[j]) x.file = files[j];
        j += 1;
      });
      if (o.held > 0) {
        const last = r.outputs.filter(x => typeof x.key === 'string' && x.key.indexOf('bgm') === 0).slice(-1)[0];
        if (last) r.outputs.push(Object.assign({}, last, { key: 'bgmheld', jp: '',
          file: '保留 ' + o.held + ' 曲（品質の基準に届かなかったため保存していません）', fileStyle: Object.assign({}, last.fileStyle, { fontFamily: "'Noto Sans JP', sans-serif", fontWeight: 700, fontSize: '12px', color: 'var(--muted)' }),
          tagStyle: Object.assign({}, last.tagStyle, { opacity: 0 }) }));
      }
    });
    // what this run did to the saved result, when it was more than a plain save: missing files were
    // rebuilt from durable state, or verified improvements replaced earlier ones (earlier ones stay in history).
    v.rows.forEach(r => {
      if (!Array.isArray(r.outputs)) return;
      ['inst', 'bgm'].forEach(t => {
        const o = (s.tst[r.key] || {})[t];
        const res = o && o.result;
        if (!res || (o.k !== 'done' && o.k !== 'reused')) return;
        const unit = t === 'bgm' ? ' 曲' : ' 件';
        const texts = [];
        if (res.state === 'REBUILT' && res.rebuilt_count > 0) texts.push('欠けていた ' + res.rebuilt_count + unit + 'を保存済みの記録から作り直しました');
        else if (res.state === 'IMPROVED_REPLACED' && res.replacement_count > 0) texts.push('検証済みの改善で ' + res.replacement_count + unit + 'を更新しました（前の版は履歴に残しています）');
        // released recordings the user can already get: identified, not extracted again
        if (t === 'bgm' && res.available_count > 0) texts.push('手元の既発売の録音と同じ ' + res.available_count + ' 曲は作っていません');
        if (t === 'bgm' && res.related_recording_count > 0) texts.push('手元の録音に関連する ' + res.related_recording_count + ' 区間も確認しました');
        // the same recording as a cue saved before (another episode or this one): numbered alike, audio untouched
        if (t === 'bgm' && res.recurring_count > 0) texts.push('前に保存した曲と同じ録音 ' + res.recurring_count + ' 曲は、同じ番号の仮題でそろえました');
        // one consensus file per recurring recording, rebuilt from every episode where it plays
        if (t === 'bgm' && res.recurrence_asset_updated_count > 0) texts.push('複数回の使用から統合版 ' + res.recurrence_asset_updated_count + ' 曲を作りました');
        if (!texts.length) return;
        const mine = r.outputs.filter(x => typeof x.key === 'string' && x.key.indexOf(t) === 0);
        const last = mine.slice(-1)[0];
        if (!last) return;
        r.outputs.splice(r.outputs.indexOf(last) + 1, 0, ...texts.map((text, k) => Object.assign({}, last, { key: t + 'change' + k, jp: '',
          file: text, fileStyle: Object.assign({}, last.fileStyle, { fontFamily: "'Noto Sans JP', sans-serif", fontWeight: 700, fontSize: '12px', color: 'var(--muted)' }),
          tagStyle: Object.assign({}, last.tagStyle, { opacity: 0 }) })));
      });
    });
    // held BGM: candidates exist but are not published. say how many when there are some.
    v.rows.forEach(r => {
      const o = (s.tst[r.key] || {}).bgm;
      if (!o || o.k !== 'held' || !(o.n > 0) || !Array.isArray(r.outputs)) return;
      r.outputs.forEach(x => {
        if (x.key === 'bgm') x.file = '候補 ' + o.n + ' 曲を保留（品質の基準に届かなかったため保存していません）';
      });
    });
    // a waveform is drawn only from measured peaks; without them the element does not exist
    v.rows.forEach(r => { if (!r.wave.length) r.hasWave = false; });
    // a validated zero result lands as SKIPPED, but nothing was separated before: say nothing untrue
    v.rows.forEach(r => { if (this.song(r.key).zero) { r.hasNote = false; r.note = ''; } });
    return v;
  }
};
