# Classification of face images by Fisher Linear Discriminant Analysis(FLDA)
<img src="https://github.com/g91358677462/FLDA-Fisher-Linear-Discriminant-Analysis-face-classification-and-recognition/blob/main/assets/FLDA%E4%BD%9C%E5%93%81%E5%9C%96.PNG" width="100%" height="100%">

資料集: 有65個不同人，每個人有21張人臉資料。(附註: 只有上傳部分資料到Github，如要完整資料請到此連結: https://drive.google.com/file/d/1IMkanT23DHAJtnjfyN3n-4cxdvMn9TlU/view?usp=sharing)

使用技術: 主成分分析（Principal components analysis, PCA）和線性判別分析(Linear discriminant analysis, LDA)。 

實驗流程:

把每個人挑出18張當作訓練資料(65*18=1170張)，剩下的當作測試資料(65*3=195張)。

PCA辨識:
  執行” OptimizeParam_pca.m”，此程式會找出最佳的PCA子空間維度數，此實驗找出最佳的子空間維度為178維，測試資料的分辨率為23.932%。

FLDA辨識:
	執行” classificationFLDA.m”，此程式從最佳的PCA子空間再從中找出最佳LDA子空間，測試資料的分辨率為81.880%。
