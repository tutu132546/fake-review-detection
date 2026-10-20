import scipy.io as sio
import numpy as np
from config import MAT_PATH

def main():
    print(f"加载文件: {MAT_PATH}")
    if not MAT_PATH.exists():
        raise FileNotFoundError(f"找不到 {MAT_PATH}，请确认是否放到了 data/raw/ 下")

    mat = sio.loadmat(str(MAT_PATH))
    
    print("\n===== 所有 key =====")
    for key in mat.keys():
        if key.startswith("__"):
            continue
        arr = mat[key]
        shape = getattr(arr, "shape", "N/A")
        dtype = getattr(arr, "dtype", "N/A")
        print(f"{key:15s} shape={str(shape):20s} dtype={dtype}")

    if "label" in mat:
        label = mat["label"].flatten()
        vals, counts = np.unique(label, return_counts=True)
        print("\n===== label 分布 =====")
        for v, c in zip(vals, counts):
            print(f"  label={v}: {c} 条 ({c/len(label)*100:.2f}%)")

if __name__ == "__main__":
    main()