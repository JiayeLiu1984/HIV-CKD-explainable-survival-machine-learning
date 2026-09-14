"""Exact count-weighted acceleration of the original patient bootstrap.

This function is installed into the original cell36 namespace. Sampling seeds,
fixed censoring references, metric definitions and output schema are unchanged.
Every point estimate and selected duplicated-patient bootstrap estimates are
validated against scikit-survival before the full 1,000-replicate computation.
"""
def run_bootstrap(common, store, metrics_df, bootstrap_values,
                  completed_replicates, checkpoint_file):
    import ast
    from numba import njit
    from sksurv.nonparametric import CensoringDistributionEstimator
    from pathlib import Path
    source_path = Path(__file__).resolve().parent.parent/'reviewer_minor_4_6_20260908/analyze.py'
    # When executed inside adapted cell36, __file__ is in reference_code.
    if not source_path.exists():
        source_path = Path(__file__).resolve().parent.parent.parent/'reviewer_minor_4_6_20260908/analyze.py'
    source = source_path.read_text(encoding='utf8')
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ['add', 'sumto', 'uno', 'auc', 'km', 'Context']]
    # get_source_segment(FunctionDef) omits decorators: restore JIT explicitly.
    code = '\n\n'.join(('' if n.name=='Context' else '@njit(nogil=True)\n')
                       + ast.get_source_segment(source, n) for n in nodes)
    code = code.replace('range(2)', 'range(self.p.shape[1])').replace('np.zeros((2,3))', 'np.zeros((self.p.shape[1],3))')
    # Re-form adjacent-score tie groups after zero-count patients are removed.
    # Otherwise a non-sampled intermediate score can incorrectly bridge a tie.
    old_auc = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name=='auc')
    exact_auc = '''def auc(count,t,e,w,order,score,horizon):
 lower=0.;num=0.;wc=0.;cc=0.;ww=0.;previous=0.;started=False
 for z in range(len(order)):
  i=order[z]
  if count[i]<=0:continue
  if started and score[i]-previous>1e-8:
   num+=ww*(lower+.5*cc);wc+=ww;lower+=cc;cc=0.;ww=0.
  started=True;previous=score[i]
  if t[i]>horizon:cc+=count[i]
  elif e[i]:ww+=count[i]*w[i]
 num+=ww*(lower+.5*cc);wc+=ww;lower+=cc
 return num/(wc*lower) if wc*lower>0 else np.nan'''
    code = code.replace(ast.get_source_segment(code,old_auc),exact_auc)
    code = code.replace('aa.append((order,st,np.r_[st[1:],n]))', 'aa.append((order,self.p[:,m,j]))')
    code = code.replace('@njit\n', '@njit(nogil=True)\n')
    ns = {'np': np, 'njit': njit, 'Surv': Surv,
          'CensoringDistributionEstimator': CensoringDistributionEstimator,
          'h': METRIC_TIMES}
    exec(compile(code, str(source_path), 'exec'), ns)
    contexts = {}; audit = []
    group_keys = [None] + list(store.group_keys)
    start = time.time()
    for landmark in PRIMARY_LANDMARK_INDICES:
        landmark = int(landmark); rows = store.rows_by_landmark[landmark]
        ns.update(t=common['analysis_time_month'][rows].astype(float),
                  e=common['event_within_60m'][rows].astype(bool),
                  local=common['local_patient_idx_long'][rows],
                  pred=np.stack([store.get(landmark, g).astype(float) for g in group_keys], axis=1))
        c = ns['Context'](np.arange(len(rows))); contexts[landmark] = c
        point = c.calc(np.ones(c.n))
        for m, g in enumerate(group_keys):
            ref = evaluate_survival_predictions(c.yy, c.yy, c.p[:, m, :])
            v = np.array([ref[k] for k in ['uno_c_index_5y', 'integrated_dynamic_auc', 'integrated_brier']])
            err = float(np.max(np.abs(v-point[m])))
            assert err < 1e-8, (landmark, g, err)
            audit.append(dict(check='full_sample', landmark=landmark, group=g, maximum_error=err))
        # Verify paired resampling with real duplicated patient records.
        rng = np.random.default_rng(RANDOM_SEED)
        sampled = rng.integers(0, EXPECTED_DEVELOPMENT_N, size=EXPECTED_DEVELOPMENT_N)
        counts = np.bincount(sampled, minlength=EXPECTED_DEVELOPMENT_N)[c.local].astype(float)
        positions = np.repeat(np.arange(c.n), counts.astype(int))
        vv = c.calc(counts)
        for m in [0, 1, len(group_keys)-1]:
            ref = evaluate_survival_predictions(c.yy, c.yy[positions], c.p[positions, m, :])
            v = np.array([ref[k] for k in ['uno_c_index_5y', 'integrated_dynamic_auc', 'integrated_brier']])
            err = float(np.max(np.abs(v-vv[m])))
            assert err < 1e-8, (landmark, group_keys[m], 'bootstrap', err)
            audit.append(dict(check='duplicated_patient_bootstrap', landmark=landmark, group=group_keys[m], maximum_error=err))
    save_json(audit, OUTPUT_DIR/'fast_metric_verification.json')
    print(f'Fast bootstrap: {len(audit)} checks against scikit-survival passed.', flush=True)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=4)
    for b in range(BOOTSTRAP_REPS):
        if completed_replicates[b]:
            continue
        rng = np.random.default_rng(RANDOM_SEED + b*1009)
        sampled = rng.integers(0, EXPECTED_DEVELOPMENT_N, size=EXPECTED_DEVELOPMENT_N, endpoint=False)
        allcounts = np.bincount(sampled, minlength=EXPECTED_DEVELOPMENT_N)
        counts_by_landmark = {l: allcounts[c.local].astype(float) for l,c in contexts.items()}
        calculated = dict(zip(contexts, pool.map(lambda item: item[1].calc(counts_by_landmark[item[0]]), contexts.items())))
        for landmark, c in contexts.items():
            count = counts_by_landmark[landmark]
            if count.sum() < 50 or count @ c.e < 5:
                continue
            values = calculated[landmark]
            for row in metrics_df.loc[metrics_df.landmark_index==landmark].itertuples(index=False):
                j = group_keys.index(str(row.group_key)); out = values[0]-values[j]
                out[2] *= -1
                change = count @ np.abs(c.p[:, j, -1]-c.p[:, 0, -1]) / count.sum()
                bootstrap_values[b, int(row.condition_position), :] = [*out, change]
        completed_replicates[b] = True
        done = int(completed_replicates.sum())
        if done % BOOTSTRAP_SAVE_EVERY == 0 or done == BOOTSTRAP_REPS:
            np.savez_compressed(checkpoint_file, bootstrap_values=bootstrap_values,
                                completed_replicates=completed_replicates)
        if done % PROGRESS_EVERY == 0 or done == BOOTSTRAP_REPS:
            print(f'Paired bootstrap {done}/{BOOTSTRAP_REPS}; elapsed {time.time()-start:.1f}s', flush=True)
    pool.shutdown()
