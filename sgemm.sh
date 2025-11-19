# use all logical threads
env OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc) MKL_DYNAMIC=0 \
python3 - <<'PY'
import numpy as np, time, os
n=4096
a=np.random.rand(n,n).astype('float32')
b=np.random.rand(n,n).astype('float32')
t=time.time(); c=a@b; dt=time.time()-t
print("All-threads SGEMM GFLOPs/s:", (2*n**3)/dt/1e9, " time:", dt, " threads:", os.cpu_count())
PY

