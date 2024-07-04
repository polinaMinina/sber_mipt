# sber_mipt

Content description:  
- notebooks:  
transactions.ipynb - get CoLES embeddings for transactions data  
geo.ipynb - get CoLES embeddings for geo data  
dialogs.ipynb - get embeddings aggregations for dialogs  
main_downstream.ipynb - concat modalities, train LGBM classifier  

- src/preprocessing:  
custom optimized PandasDataPreprocessor class  

- Trx aggregations and dialogs.ipynb - improved baseline (additional aggregations and dialogs added)

Best result:  
public score 0.6244 gini  
