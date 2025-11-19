# Still inside the env
PY=$(which python)

sudo perf stat -x, --no-big-num -a \
  -e uncore_imc_free_running/data_read/,uncore_imc_free_running/data_write/ \
  -- "$PY" - <<'PYCODE'
import numpy as np, time

n = 150_000_000   # ~600 MB float32; adjust if RAM is tight
a = np.random.rand(n).astype(np.float32)
b = np.empty_like(a)

t0 = time.time()
b[:] = a
dt = time.time() - t0

print("copy seconds:", dt)
PYCODE

