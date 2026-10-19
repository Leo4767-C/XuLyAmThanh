"""
Lab 3 - Nhan dang tieng noi nang cao voi HMM (CSE457)
Sinh vien: Le Van Chuong - MSSV: 2351260643
Chay:  python Lab03.py            (can thu muc dataset/train, dataset/test)
Tao ra: models/, figures/*.png, results.txt, lexicon.txt, corpus.txt
"""
import time, pickle, math, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

warnings.filterwarnings("ignore")
SR = 16000
DATA = Path("dataset")
LABELS = ["mo", "dong", "len", "xuong", "dung"]
Path("figures").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
LOG = open("results.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


# ---------------------------------------------------------------- A. Features
def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)          # endpoint detection
    return librosa.effects.preemphasis(y, coef=0.97)


def feat_from_signal(y, dynamic=True):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=512,
                                win_length=int(0.025 * SR), hop_length=int(0.010 * SR),
                                window="hamming", n_mels=26, fmin=20, fmax=SR / 2)
    if dynamic:
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        F = np.vstack([mfcc, d1, d2]).T                 # T x 39
    else:
        F = mfcc.T                                      # T x 13
    mu = F.mean(0, keepdims=True)                       # CMVN theo utterance (khong dung thong ke test cho train)
    sd = F.std(0, keepdims=True) + 1e-8
    return (F - mu) / sd


def load_split(split, dynamic=True, noise_snr=None, seed=0):
    rng = np.random.default_rng(seed)
    X, y, names = [], [], []
    for lab in LABELS:
        for wav in sorted((DATA / split / lab).glob("*.wav")):
            sig = load_wav(str(wav))
            if noise_snr is not None:                   # E5: them nhieu trang muc SNR cho truoc
                p = np.mean(sig ** 2)
                n = rng.normal(0, np.sqrt(p / 10 ** (noise_snr / 10)), len(sig))
                sig = sig + n
            X.append(feat_from_signal(sig, dynamic)); y.append(lab); names.append(wav.name)
    return X, y, names


def plot_heatmaps(word_list=("mo", "dong")):
    fig, ax = plt.subplots(len(word_list), 3, figsize=(13, 3 * len(word_list)))
    for r, w in enumerate(word_list):
        wav = sorted((DATA / "train" / w).glob("*.wav"))[0]
        F = feat_from_signal(load_wav(str(wav)))
        for c, (nm, sl) in enumerate([("MFCC", slice(0, 13)), ("Delta", slice(13, 26)), ("Delta2", slice(26, 39))]):
            ax[r, c].imshow(F[:, sl].T, aspect="auto", origin="lower", cmap="viridis")
            ax[r, c].set_title(f"{w}: {nm}"); ax[r, c].set_xlabel("frame")
    plt.tight_layout(); plt.savefig("figures/heatmaps.png", dpi=130); plt.close()


# ---------------------------------------------------------------- B. HMM L2R
class LRGaussianHMM(GaussianHMM):
    """GaussianHMM + giu state cuoi la absorbing (a_NN=1) sau moi M-step.
    hmmlearn khong quan sat chuyen tiep ra khoi state cuoi -> hang cuoi cua transmat_ = 0/NaN."""
    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        A = np.nan_to_num(self.transmat_)
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        for i in range(len(A) - 1):                       # hang chua du tong -> chuan hoa lai
            s = A[i].sum()
            if s <= 0:
                A[i, i], A[i, i + 1] = 0.7, 0.3
            else:
                A[i] /= s
        self.transmat_ = A


def make_lr_hmm(n_states=5, n_iter=80, cov="diag"):
    # init_params="" vi means/covars duoc khoi tao thu cong bang uniform segmentation (flat start)
    m = LRGaussianHMM(n_components=n_states, covariance_type=cov, n_iter=n_iter, tol=1e-3,
                      min_covar=1e-3, init_params="", params="tmc", random_state=0)
    start = np.zeros(n_states); start[0] = 1.0
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i], A[i, i + 1] = 0.7, 0.3
    A[-1, -1] = 1.0
    m.startprob_, m.transmat_ = start, A
    return m


def check_topology(m):
    A = m.transmat_
    n = len(A)
    bad = [(i, j) for i in range(n) for j in range(n) if (j < i or j > i + 1) and A[i, j] > 1e-8]
    return bad


# ---------------------------------------------------------------- C. Training
def flat_start(m, feats, cov="diag", floor=1e-2):
    """Chia deu moi utterance thanh N doan lien tiep -> mean/var ban dau cua tung state.
    Dam bao state i ung voi phan i cua tu (dung thu tu trai->phai), khac K-means (khong co thu tu)."""
    n = m.n_components
    segs = [[] for _ in range(n)]
    for F in feats:
        for i, part in enumerate(np.array_split(F, n)):
            segs[i].append(part)
    D = feats[0].shape[1]
    means = np.zeros((n, D))
    covars = np.zeros((n, D)) if cov == "diag" else np.zeros((n, D, D))
    for i in range(n):
        S = np.vstack(segs[i])
        means[i] = S.mean(0)
        if cov == "diag":
            covars[i] = np.maximum(S.var(0), floor)
        else:
            covars[i] = np.cov(S.T) + floor * np.eye(D) if len(S) > D else np.eye(D)
    m.means_ = means
    m.covars_ = covars



def train_models(Xtr, ytr, n_states=5, dynamic=True, cov="diag", verbose=False):
    models = {}
    for lab in LABELS:
        feats = [f for f, l in zip(Xtr, ytr) if l == lab]
        X = np.vstack(feats); lengths = [len(f) for f in feats]
        m = make_lr_hmm(n_states, cov=cov)
        flat_start(m, feats, cov)
        m.fit(X, lengths)
        models[lab] = m
        if verbose:
            log(f"[{lab}] converged={m.monitor_.converged}  bad_transitions={check_topology(m)}")
            log(np.round(m.transmat_, 3))
    return models


# ---------------------------------------------------------------- D. Recognition
def recognize(F, models):
    scores = {k: m.score(F) for k, m in models.items()}
    return max(scores, key=scores.get), scores


def evaluate(models, X, y):
    pred = [recognize(F, models)[0] for F in X]
    return accuracy_score(y, pred), pred


def save_cm(y, pred, title, fname):
    cm = confusion_matrix(y, pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title(title); plt.savefig(fname, dpi=130); plt.close()
    return cm


def viterbi_plots(models, Xte, yte, names, k=2):
    fig, ax = plt.subplots(1, k, figsize=(6 * k, 3.5))
    for i in range(k):
        F, lab = Xte[i * 2], yte[i * 2]                 # moi tu 1 utterance dau
        lp, st = models[lab].decode(F, algorithm="viterbi")
        ax[i].step(range(len(st)), st, where="post")
        ax[i].set_title(f"Viterbi: {names[i*2]} vs model {lab}"); ax[i].set_xlabel("frame"); ax[i].set_ylabel("state")
        dur = np.bincount(st, minlength=models[lab].n_components)
        log(f"Viterbi {names[i*2]}: logp={lp:.1f}, duration/state={dur.tolist()}, monotone={bool(np.all(np.diff(st) >= 0))}")
    plt.tight_layout(); plt.savefig("figures/viterbi.png", dpi=130); plt.close()


# ---------------------------------------------------------------- E. DTW baseline
def dtw_cost(A, B):
    D, wp = librosa.sequence.dtw(X=A.T, Y=B.T, metric="euclidean")
    return D[-1, -1] / len(wp)


def dtw_predict(F, Xtr, ytr):
    best, lab = np.inf, None
    for T, l in zip(Xtr, ytr):
        c = dtw_cost(F, T)
        if c < best:
            best, lab = c, l
    return lab


# ---------------------------------------------------------------- F. Lexicon
def write_lexicon():
    # Phone inventory CHI MANG TINH MINH HOA - thay bang bo phone cua giang vien.
    lex = {"mo": "m o", "dong": "d o ng", "len": "l e n", "xuong": "x uo ng", "dung": "d u ng"}
    with open("lexicon.txt", "w", encoding="utf-8") as f:
        for w, p in lex.items():
            f.write(f"{w} {p}\n")


def read_lexicon(path="lexicon.txt"):
    d = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if parts:
            d[parts[0]] = parts[1:]
    return d


# ---------------------------------------------------------------- G. Bigram
CORPUS = ["mo len", "mo dong", "len dung", "xuong dung", "dong len", "len xuong",
          "xuong dong", "dung mo", "mo len dung", "dong xuong dung", "len dung", "xuong dung"]


def build_counts(corpus):
    c1, c2 = Counter(), Counter()
    for s in corpus:
        w = ["<s>"] + s.split()
        c1.update(w[:-1])
        c2.update(zip(w[:-1], w[1:]))
    return c1, c2


def bigram_prob(prev, cur, c2, c1, V, k=0.5):
    return (c2.get((prev, cur), 0) + k) / (c1.get(prev, 0) + k * V)


def rerank(F, prev, models, c2, c1, lm_weight):
    T = len(F)
    tot = {w: m.score(F) / T + lm_weight * math.log(bigram_prob(prev, w, c2, c1, len(models)))
           for w, m in models.items()}
    return max(tot, key=tot.get), tot


# ---------------------------------------------------------------- Main
def main():
    log("=== A. Feature extraction ===")
    t0 = time.time()
    Xtr, ytr, ntr = load_split("train")
    Xte, yte, nte = load_split("test")
    log("shapes:", [F.shape for F in Xtr[:3]], " (chieu 2 phai = 39)")
    log(f"train={len(Xtr)} test={len(Xte)}  leakage={set(ntr) & set(nte) != set() and 'CHECK'}")
    plot_heatmaps()

    log("\n=== B,C. Train HMM 5 state, diag ===")
    t0 = time.time()
    models = train_models(Xtr, ytr, 5, verbose=True)
    t_hmm_train = time.time() - t0
    for k, m in models.items():
        pickle.dump(m, open(f"models/{k}.pkl", "wb"))

    log("\n=== D. Recognition ===")
    t0 = time.time()
    acc_te, pred_te = evaluate(models, Xte, yte)
    t_hmm_test = (time.time() - t0) / len(Xte)
    acc_tr, _ = evaluate(models, Xtr, ytr)
    log(f"HMM train acc={acc_tr:.3f}  test acc={acc_te:.3f}")
    cm_hmm = save_cm(yte, pred_te, "HMM confusion (test)", "figures/cm_hmm.png")
    log("score table (3 test utterances):")
    for i in [0, len(Xte) // 2, len(Xte) - 1]:
        p, sc = recognize(Xte[i], models)
        srt = sorted(sc.items(), key=lambda z: -z[1])
        log(f"  {nte[i]} (true={yte[i]}, pred={p}, top1-top2 gap={srt[0][1]-srt[1][1]:.1f})")
        for k, v in srt:
            log(f"     {k:6s} {v:12.2f}")
    viterbi_plots(models, Xte, yte, nte)

    log("\n=== E1. n_states in {3,5,7} ===")
    for n in (3, 5, 7):
        ms = train_models(Xtr, ytr, n)
        a, _ = evaluate(ms, Xte, yte)
        conv = sum(m.monitor_.converged for m in ms.values())
        log(f"n_states={n}: test acc={a:.3f}  converged={conv}/5")

    log("\n=== E2. MFCC13 vs MFCC+d+d2 ===")
    X13tr, _, _ = load_split("train", dynamic=False)
    X13te, _, _ = load_split("test", dynamic=False)
    a13, p13 = evaluate(train_models(X13tr, ytr, 5, dynamic=False), X13te, yte)
    log(f"13-D acc={a13:.3f}   39-D acc={acc_te:.3f}")
    save_cm(yte, p13, "HMM 13-D confusion", "figures/cm_hmm13.png")

    log("\n=== E3. diag vs full covariance ===")
    try:
        mf = train_models(Xtr, ytr, 5, cov="full")
        log(f"full: train={evaluate(mf, Xtr, ytr)[0]:.3f} test={evaluate(mf, Xte, yte)[0]:.3f}  | "
            f"diag: train={acc_tr:.3f} test={acc_te:.3f}")
    except Exception as e:
        log("full covariance failed:", e)

    log("\n=== E4. DTW vs HMM ===")
    t0 = time.time()
    pred_dtw = [dtw_predict(F, Xtr, ytr) for F in Xte]
    t_dtw_test = (time.time() - t0) / len(Xte)
    acc_dtw = accuracy_score(yte, pred_dtw)
    cm_dtw = save_cm(yte, pred_dtw, "DTW confusion (test)", "figures/cm_dtw.png")

    def top_conf(cm):
        c = cm.copy(); np.fill_diagonal(c, 0)
        i, j = np.unravel_index(c.argmax(), c.shape)
        return f"{LABELS[i]}->{LABELS[j]} ({c[i, j]})" if c[i, j] else "none"
    log("| Metric | DTW | HMM |")
    log(f"| Accuracy | {acc_dtw:.3f} | {acc_te:.3f} |")
    log(f"| Train samples/class | {len(Xtr)//5} templates | {len(Xtr)//5} utt (tham so A,B,pi) |")
    log(f"| Train time | ~0 s | {t_hmm_train:.1f} s |")
    log(f"| Test time/trial | {t_dtw_test*1000:.1f} ms | {t_hmm_test*1000:.1f} ms |")
    log(f"| Top confusion | {top_conf(cm_dtw)} | {top_conf(cm_hmm)} |")
    # DTW voi it mau: 1 template/lop
    first = {}
    for F, l in zip(Xtr, ytr):
        first.setdefault(l, F)
    p1 = [dtw_predict(F, list(first.values()), list(first.keys())) for F in Xte]
    log(f"DTW 1 template/class acc={accuracy_score(yte, p1):.3f}")
    # HMM voi 2 utt/lop
    sub_X, sub_y, cnt = [], [], Counter()
    for F, l in zip(Xtr, ytr):
        if cnt[l] < 2:
            sub_X.append(F); sub_y.append(l); cnt[l] += 1
    try:
        a2, _ = evaluate(train_models(sub_X, sub_y, 3), Xte, yte)
        log(f"HMM 2 utt/class (3 states) acc={a2:.3f}")
    except Exception as e:
        log("HMM 2 utt/class failed:", e)

    log("\n=== E5. Mismatch: noise SNR ===")
    for snr in (20, 10, 5):
        Xn, yn, _ = load_split("test", noise_snr=snr)
        an, _ = evaluate(models, Xn, yn)
        ad = accuracy_score(yn, [dtw_predict(F, Xtr, ytr) for F in Xn])
        log(f"SNR={snr} dB: HMM={an:.3f}  DTW={ad:.3f}  (clean HMM={acc_te:.3f}, DTW={acc_dtw:.3f})")

    log("\n=== F. Lexicon ===")
    write_lexicon()
    log(read_lexicon())

    log("\n=== G / E6. Bigram re-ranking ===")
    with open("corpus.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(CORPUS))
    c1, c2 = build_counts(CORPUS)
    # chon utterance co top1-top2 gap nho nhat
    gaps = []
    for i, F in enumerate(Xte):
        _, sc = recognize(F, models)
        s = sorted(sc.values(), reverse=True)
        gaps.append((s[0] - s[1]) / len(F))
    i = int(np.argmin(gaps))
    log(f"utterance gan nhat: {nte[i]} true={yte[i]}  gap/frame={gaps[i]:.3f}")
    for prev in ["<s>"] + LABELS:
        row = []
        for lw in (0, 0.1, 0.3, 1.0, 3.0):
            row.append(f"{lw}:{rerank(Xte[i], prev, models, c2, c1, lw)[0]}")
        log(f"  prev={prev:6s} " + "  ".join(row))

    log("\n=== H. Score distribution & reject ===")
    right, wrong, gapl = [], [], []
    for F, l in zip(Xte, yte):
        _, sc = recognize(F, models)
        T = len(F)
        right.append(sc[l] / T)
        wrong += [v / T for k, v in sc.items() if k != l]
        s = sorted(sc.values(), reverse=True); gapl.append((s[0] - s[1]) / T)
    plt.hist(right, bins=15, alpha=.6, label="dung lop"); plt.hist(wrong, bins=15, alpha=.6, label="sai lop")
    plt.xlabel("log-likelihood / frame"); plt.legend(); plt.savefig("figures/score_dist.png", dpi=130); plt.close()
    thr = np.percentile(right, 5)
    log(f"reject threshold de xuat (5th pct score/frame cua dung lop) = {thr:.2f}; "
        f"mean dung={np.mean(right):.2f}, mean sai={np.mean(wrong):.2f}")
    log("Co the ket hop them: reject neu gap top1-top2 per frame < nguong nho.")
    LOG.close()


if __name__ == "__main__":
    main()
