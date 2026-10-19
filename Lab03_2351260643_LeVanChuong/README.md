# Lab 3 - Nhan dang tieng noi voi Gaussian HMM (CSE457)
**Sinh viên:** Lê Văn Chương | **MSSV:** 2351260643

## Cau truc
- `Lab03.py`              : pipeline day du (A->H, thi nghiem E1-E6). Ket qua ghi vao `results.txt`, `figures/`, `models/`.
- `Lab03_Notebook.ipynb`  : Jupyter Notebook tuong tac, chay toan bo cac buoc voi bieu do inline va giai thich ly thuyet.
- `lexicon.txt`, `corpus.txt` : lexicon mini va corpus bigram.
- `figures/`, `models/`   : cac bieu do truc quan hoa va mo hinh da huan luyen.
- `Lab03_Report.docx`     : bao cao thuc hanh chi tiet (dinh dang Word).

## Cach chay
1. Cai dat thu vien:
   ```bash
   pip install numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn
   ```
2. Chay pipeline:
   ```bash
   python3 Lab03.py
   ```
   hoac mo va chay file `Lab03_Notebook.ipynb` trong Jupyter Notebook / VS Code.

## Luu y
- hmmlearn khong quan sat chuyen tiep ra khoi state cuoi -> hang cuoi transmat_ thanh 0/NaN. Lop `LRGaussianHMM`
  ep a_NN=1 sau moi M-step; khoi tao means/covars bang uniform segmentation (flat start) de giu thu tu trai->phai.
- Ket qua trong `results_synthetic.txt` KHONG phai ket qua tren du lieu giong that; hay chay lai tren du lieu cua ban
  va thay so lieu vao bao cao.
